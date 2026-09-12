"""Read account-scoped email from Apple Mail's local message store."""

from __future__ import annotations

import email as email_lib
import json
import plistlib
import re
import sqlite3
from datetime import datetime, timezone
from email.utils import getaddresses
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import unquote

from openjarvis.connectors._stubs import BaseConnector, Document, SyncStatus
from openjarvis.connectors.gmail_imap import (
    _decode_header,
    _extract_text_body,
    _parse_date,
)
from openjarvis.core.config import DEFAULT_CONFIG_DIR
from openjarvis.core.registry import ConnectorRegistry

_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "connectors" / "apple_mail.json"
_EXCLUDED_MAILBOXES = ("Deleted Items", "Drafts", "Junk Email", "Outbox")
_MAIL_VERSION = re.compile(r"^V(\d+)$")


class AppleMailStoreError(RuntimeError):
    """The local Apple Mail store could not be read.

    Raised instead of returning empty results so a failed read is never
    mistaken for a store that legitimately holds nothing: an empty sync
    completes and advances the watermark, a raised error does not.
    """


def _mail_store_versions(mail_root: Path) -> list[Path]:
    """Return the ``V*`` stores newest first, ordered numerically.

    Sorting the names lexicographically puts ``V9`` above ``V10``, which would
    read a superseded store; ``V10`` is the tenth format, not the first.
    """

    def key(path: Path) -> tuple[int, str]:
        match = _MAIL_VERSION.match(path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return sorted(mail_root.glob("V*"), key=key, reverse=True)


def _aware_timestamp(msg) -> datetime:
    """Return the message date as an aware UTC timestamp.

    ``_parse_date`` falls back to a naive ``datetime.now()`` for a missing or
    malformed ``Date``, and ``parsedate_to_datetime`` itself returns a naive
    value for a header carrying no usable zone (RFC 5322 ``-0000``). Either one
    would raise when compared with the aware ``since`` cutoff, so a tz-less
    value is read as UTC.
    """
    timestamp = _parse_date(msg)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _received_timestamp(plist: dict[str, Any]) -> Optional[datetime]:
    """Return the arrival time Mail.app wrote beside the message, if any.

    ``date-received`` is Mail.app's own record of when this message reached
    this store, so it is what an incremental sync asks about. A trailer that
    omits it, or carries something that is not a usable epoch, yields ``None``
    and the caller falls back to the sender's ``Date`` header.
    """
    received = plist.get("date-received")
    if isinstance(received, bool) or not isinstance(received, (int, float)):
        return None
    if received <= 0:
        return None
    try:
        return datetime.fromtimestamp(received, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


_ADDRESS_SPECIALS = re.compile(r'[][\\()<>@,:;".]')


def _format_mailbox(name: str, address: str) -> str:
    """Put one parsed recipient back together as a single readable entry.

    ``email.utils.formataddr`` would re-encode a non-ASCII display name into an
    encoded word, undoing the decoding this connector just did, so the quoting
    happens here: a name carrying a character the header grammar treats as
    special is quoted, and a backslash or quote inside it is escaped. For an
    ASCII name the result is byte-for-byte what ``formataddr`` produces.
    """
    if not name:
        return address
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    if _ADDRESS_SPECIALS.search(name):
        escaped = f'"{escaped}"'
    return f"{escaped} <{address}>"


def _split_address_list(raw: object) -> list[str]:
    """Return one entry per recipient the header names.

    Splitting on commas turns ``"Doe, Jane" <jane@example.test>`` into two
    fragments, neither of them a recipient. ``getaddresses`` reads the header's
    own grammar instead, so a quoted display name that contains a comma stays
    one recipient. The header is parsed before any encoded word is decoded,
    because the grammar is written in that encoded form; each display name is
    decoded afterwards, so a reader sees the name rather than its encoding.
    """
    text = raw if isinstance(raw, str) else str(raw or "")
    entries: list[str] = []
    for display, address in getaddresses([text]):
        name = _decode_header(display).strip()
        mailbox = address.strip()
        if not name and not mailbox:
            continue
        entries.append(_format_mailbox(name, mailbox))
    return entries


def _mailbox_label(url: str, account_id: str) -> str:
    """Return the mailbox name the index's URL stands for.

    The stored value is a URL such as ``imap://ACCOUNT/INBOX/Saved%20Mail``,
    which is not something to put in front of a reader. Dropping the scheme
    and the account segment, percent-decoding each remaining segment and
    rejoining on ``/`` keeps a nested mailbox nested and makes an escaped
    character readable. A URL with nothing left after the account yields an
    empty name rather than a raw URL wearing a label's clothes.
    """
    remainder = url.split("://", 1)[-1]
    segments = [segment for segment in remainder.split("/") if segment]
    if segments and segments[0] == account_id:
        segments = segments[1:]
    return "/".join(unquote(segment) for segment in segments)


@ConnectorRegistry.register("apple_mail")
class AppleMailConnector(BaseConnector):
    """Index locally cached Apple Mail messages for one configured account."""

    connector_id = "apple_mail"
    display_name = "Apple Mail"
    auth_type = "local"

    def __init__(
        self,
        config_path: str = "",
        mail_root: str = "",
        *,
        max_messages: int = 500,
    ) -> None:
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._mail_root = (
            Path(mail_root) if mail_root else Path.home() / "Library" / "Mail"
        )
        self._max_messages = max_messages
        self._items_synced = 0
        self._items_total = 0
        self._last_sync: Optional[datetime] = None

    def _config(self) -> dict[str, str]:
        try:
            value = json.loads(self._config_path.read_text())
        except (OSError, ValueError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _paths_for(self, account_id: str) -> tuple[Path, Path] | None:
        if not account_id:
            return None
        for version in _mail_store_versions(self._mail_root):
            account_root = version / account_id
            db_path = version / "MailData" / "Envelope Index"
            if account_root.is_dir() and db_path.is_file():
                return account_root, db_path
        return None

    def _paths(self) -> tuple[Path, Path] | None:
        return self._paths_for(str(self._config().get("account_id", "")).strip())

    @staticmethod
    def _mailbox_schemes(db_path: Path, account_id: str) -> list[str]:
        """Return the distinct URL schemes Apple Mail recorded for an account."""
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT mb.url FROM mailboxes mb WHERE mb.url LIKE ?",
                    (f"%://{account_id}/%",),
                )
                schemes = {str(row[0]).split("://", 1)[0] for row in rows}
        except sqlite3.Error as exc:
            raise AppleMailStoreError(
                f"Could not read the Apple Mail index at {db_path}: {exc}"
            ) from exc
        return sorted(scheme for scheme in schemes if scheme)

    def available_accounts(self) -> list[dict[str, str]]:
        """List local Apple Mail accounts without reading any message body.

        An unreadable index raises rather than returning an empty list: a
        missing Full Disk Access grant must not read as "this Mac has no mail
        accounts", which is what the caller would otherwise be told.
        """
        accounts: list[dict[str, str]] = []
        seen: set[str] = set()
        for version in _mail_store_versions(self._mail_root):
            db_path = version / "MailData" / "Envelope Index"
            if not db_path.is_file():
                continue
            for account_root in sorted(version.iterdir()):
                account_id = account_root.name
                if not account_root.is_dir() or account_id in seen:
                    continue
                if account_id == "MailData" or account_id.startswith("."):
                    continue
                schemes = self._mailbox_schemes(db_path, account_id)
                if not schemes:
                    continue
                seen.add(account_id)
                accounts.append(
                    {
                        "account_id": account_id,
                        "protocol": ",".join(schemes),
                        "mail_version": version.name,
                    }
                )
        return accounts

    def configure(self, account_id: str) -> None:
        """Pin this connector to one local account, rejecting unusable ones.

        Repinning a connected connector to a *different* account is refused
        rather than performed. The indexed mail and the sync checkpoint belong
        to the account currently pinned, and only the disconnect lifecycle
        purges them; overwriting the id here would leave the previous account's
        mail indexed under this connector and start the new account at the old
        account's watermark, skipping everything older.
        """
        account_id = (account_id or "").strip()
        if not account_id:
            raise ValueError("An Apple Mail account id is required")
        current = str(self._config().get("account_id", "")).strip()
        if current and current != account_id:
            raise ValueError(
                f"Apple Mail is already connected to account {current!r}. "
                f"Disconnect it first — that stops the sync, purges the mail it "
                f"indexed and resets its checkpoint — then connect "
                f"{account_id!r}."
            )
        known = {
            account["account_id"]: account for account in self.available_accounts()
        }
        if account_id not in known:
            available = ", ".join(sorted(known)) or "none"
            raise ValueError(
                f"Unknown Apple Mail account {account_id!r}. "
                f"Locally available accounts: {available}"
            )
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps(
                {
                    "account_id": account_id,
                    "protocol": known[account_id]["protocol"],
                },
                indent=2,
            )
        )

    def is_connected(self) -> bool:
        """Whether an account is pinned and its local store is present.

        This is a cheap presence check, not a read probe: whether the index is
        actually readable is answered by ``sync``, which raises rather than
        reporting a successful empty run.
        """
        return self._paths() is not None

    def disconnect(self) -> None:
        self._config_path.unlink(missing_ok=True)

    @staticmethod
    def _read_emlx(path: Path):
        """Parse an ``.emlx`` file into its message and its trailing plist.

        The format is ``<byte-count>\\n<byte-count bytes of RFC822><plist>``;
        reading past the count folds Apple's trailing property list into the
        message body.

        A trailer that is absent, empty or unparseable yields an empty
        mapping rather than dropping the file. The trailer is Mail.app's own
        bookkeeping about the message; the message itself still reads, and
        losing metadata is a poor reason to lose mail.
        """
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        prefix, separator, remainder = raw.partition(b"\n")
        if not separator:
            return None
        try:
            length = int(prefix.strip())
        except ValueError:
            return None
        if length < 0:
            return None
        plist: dict[str, Any] = {}
        trailer = remainder[length:]
        if trailer.strip():
            try:
                loaded = plistlib.loads(trailer)
            except Exception:  # noqa: BLE001 - any malformed trailer is skipped
                loaded = None
            if isinstance(loaded, dict):
                plist = loaded
        return email_lib.message_from_bytes(remainder[:length]), plist

    def sync(
        self,
        *,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
    ) -> Iterator[Document]:
        del cursor
        paths = self._paths()
        if paths is None:
            return
        account_root, db_path = paths
        account_id = str(self._config().get("account_id", "")).strip()
        query = (
            "SELECT m.ROWID, mb.url FROM messages m "
            "JOIN mailboxes mb ON mb.ROWID=m.mailbox "
            "WHERE mb.url LIKE ? AND m.deleted=0 "
            + "".join("AND mb.url NOT LIKE ? " for _ in _EXCLUDED_MAILBOXES)
            + "ORDER BY m.date_received DESC"
        )
        params = [
            # Any protocol Apple Mail records for the account: ews://, imap://,
            # pop://, local://. The messages themselves are read from .emlx
            # files, which every account type writes identically.
            f"%://{account_id}/%",
            *(f"%/{name}" for name in _EXCLUDED_MAILBOXES),
        ]
        if self._max_messages > 0:
            query += " LIMIT ?"
            params.append(self._max_messages)

        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                # The mailbox name travels with the row, so it comes from the
                # account's own index rather than from the file's path.
                mailboxes = {
                    str(row[0]): _mailbox_label(str(row[1] or ""), account_id)
                    for row in conn.execute(query, params)
                }
        except sqlite3.Error as exc:
            # Returning here would end the generator normally, which SyncEngine
            # records as a completed sync: the error would be cleared and the
            # watermark advanced past mail that was never read.
            raise AppleMailStoreError(
                f"Could not read the Apple Mail index at {db_path}: {exc}"
            ) from exc

        self._items_total = len(mailboxes)
        # A message Mail.app has not finished downloading is stored as
        # <row>.partial.emlx, so the row id is the name up to its first dot
        # rather than the stem. When both files exist for one row the
        # complete one wins, whichever order the walk reaches them in.
        paths_by_id: dict[str, Path] = {}
        for path in sorted(account_root.rglob("*.emlx")):
            emlx_uid = path.name.split(".", 1)[0]
            if emlx_uid not in mailboxes:
                continue
            current = paths_by_id.get(emlx_uid)
            if current is None or (
                current.name.endswith(".partial.emlx")
                and not path.name.endswith(".partial.emlx")
            ):
                paths_by_id[emlx_uid] = path
        synced = 0
        for row_id, mailbox in mailboxes.items():
            path = paths_by_id.get(row_id)
            parsed = self._read_emlx(path) if path is not None else None
            if parsed is None:
                continue
            msg, plist = parsed
            timestamp = _received_timestamp(plist) or _aware_timestamp(msg)
            if since is not None:
                cutoff = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                if timestamp < cutoff:
                    continue
            header_message_id = _decode_header(msg.get("Message-ID", "")).strip()
            message_id = header_message_id or row_id
            recipients = _split_address_list(msg.get("To", "")) + _split_address_list(
                msg.get("Cc", "")
            )
            partial = path.name.endswith(".partial.emlx")
            synced += 1
            yield Document(
                doc_id=f"apple_mail:{account_id}:{message_id}",
                source="apple_mail",
                doc_type="email",
                content=_extract_text_body(msg),
                title=_decode_header(msg.get("Subject", "")),
                author=_decode_header(msg.get("From", "")),
                participants=recipients,
                timestamp=timestamp,
                thread_id=_decode_header(msg.get("In-Reply-To", "")) or None,
                metadata={
                    "account_id": account_id,
                    "local_row_id": row_id,
                    "message_id": header_message_id,
                    "mailbox": mailbox,
                    "emlx_uid": row_id,
                    "partial": partial,
                },
                source_id=header_message_id or f"{account_id}:{mailbox}:{row_id}",
                participants_raw=recipients,
                channel=mailbox or None,
            )
        self._items_synced = synced
        self._last_sync = datetime.now(tz=timezone.utc)

    def sync_status(self) -> SyncStatus:
        return SyncStatus(
            state="idle",
            items_synced=self._items_synced,
            items_total=self._items_total,
            last_sync=self._last_sync,
        )
