from __future__ import annotations

import json
import plistlib
import sqlite3
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pytest

from openjarvis.connectors import apple_mail as apple_mail_module
from openjarvis.connectors.apple_mail import AppleMailConnector
from openjarvis.connectors.pipeline import IngestionPipeline
from openjarvis.connectors.store import KnowledgeStore
from openjarvis.connectors.sync_engine import SyncEngine

# Apple appends a property list after the declared RFC822 byte count; reading
# past the count folds it into the message body.
_TRAILING_PLIST = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<plist version="1.0"><dict><key>flags</key><integer>1</integer>'
    b"</dict></plist>\n"
)


_DEFAULT_DATE = "Mon, 01 Sep 2025 12:00:00 +0000"
_OLD_DATE = "Wed, 01 Jan 2020 00:00:00 +0000"

# Mail.app records its own arrival time in the trailer as a Unix epoch.
_RECEIVED_EPOCH = 1789000000
_RECEIVED_AT = datetime(2026, 9, 10, 0, 26, 40, tzinfo=timezone.utc)
_OLD_EPOCH = 1577836800
_CUTOFF = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _store_error() -> type[Exception]:
    """Read the error class back out of the live module.

    The server registers connectors by reloading every ``openjarvis.connectors``
    module, and a reload rebuilds the classes inside the module it already
    shares with this file. A class captured at import time is then a different
    object from the one a connector raises, so ``pytest.raises`` stops matching
    once any test in the same process has touched that registration path.
    """
    return apple_mail_module.AppleMailStoreError


def _message(
    subject: str = "Expected subject",
    body: str = "Expected body",
    *,
    date_header: str | None = _DEFAULT_DATE,
) -> bytes:
    msg = EmailMessage()
    msg["Message-ID"] = "<expected@example.test>"
    msg["Subject"] = subject
    msg["From"] = "sender@example.test"
    msg["To"] = "jfaber@caltech.edu"
    if date_header is not None:
        msg["Date"] = date_header
    msg.set_content(body)
    return msg.as_bytes()


def _fixture(
    tmp_path: Path,
    *,
    scheme: str = "ews",
    account_id: str = "ACCOUNT-ID",
    write_config: bool = True,
) -> tuple[Path, Path]:
    root = tmp_path / "Mail"
    version = root / "V10"
    account = version / account_id / "Inbox.mbox" / "Data" / "Messages"
    account.mkdir(parents=True)
    db = version / "MailData" / "Envelope Index"
    db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE mailboxes (ROWID INTEGER PRIMARY KEY, url TEXT);"
            "CREATE TABLE messages (ROWID INTEGER PRIMARY KEY, mailbox INTEGER, "
            "deleted INTEGER, date_received INTEGER);"
            f"INSERT INTO mailboxes VALUES (1, '{scheme}://{account_id}/Inbox');"
            "INSERT INTO messages VALUES (42, 1, 0, 2);"
        )
    raw = _message()
    (account / "42.emlx").write_bytes(
        str(len(raw)).encode() + b"\n" + raw + _TRAILING_PLIST
    )
    config = tmp_path / "apple_mail.json"
    if write_config:
        config.write_text(json.dumps({"account_id": account_id}))
    return root, config


def _write_message(
    root: Path,
    account_id: str,
    row_id: str,
    *,
    version: str = "V10",
    subject: str = "Expected subject",
    date_header: str | None = _DEFAULT_DATE,
) -> None:
    messages = root / version / account_id / "Inbox.mbox" / "Data" / "Messages"
    messages.mkdir(parents=True, exist_ok=True)
    raw = _message(subject, date_header=date_header)
    (messages / f"{row_id}.emlx").write_bytes(
        str(len(raw)).encode() + b"\n" + raw + _TRAILING_PLIST
    )


def _add_account(
    root: Path,
    account_id: str,
    *,
    scheme: str = "imap",
    version: str = "V10",
    subject: str = "Expected subject",
) -> None:
    """Add a second synthetic account, creating its store if it is new."""
    db = root / version / "MailData" / "Envelope Index"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS mailboxes "
            "(ROWID INTEGER PRIMARY KEY, url TEXT);"
            "CREATE TABLE IF NOT EXISTS messages (ROWID INTEGER PRIMARY KEY, "
            "mailbox INTEGER, deleted INTEGER, date_received INTEGER);"
        )
        mailbox_id = (
            conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM mailboxes").fetchone()[0]
            + 1
        )
        row_id = (
            conn.execute("SELECT COALESCE(MAX(ROWID), 41) FROM messages").fetchone()[0]
            + 1
        )
        conn.execute(
            "INSERT INTO mailboxes VALUES (?, ?)",
            (mailbox_id, f"{scheme}://{account_id}/Inbox"),
        )
        conn.execute("INSERT INTO messages VALUES (?, ?, 0, 2)", (row_id, mailbox_id))
    _write_message(root, account_id, str(row_id), version=version, subject=subject)


def _corrupt_index(root: Path, version: str = "V10") -> None:
    """Make the Envelope Index unreadable the way a denied grant does."""
    (root / version / "MailData" / "Envelope Index").write_bytes(b"not a database")


def _raw_message(
    *,
    subject: str = "Expected subject",
    body: str = "Expected body",
    date_header: str | None = _DEFAULT_DATE,
    message_id: str | None = "<expected@example.test>",
    to: str = "jfaber@caltech.edu",
    cc: str | None = None,
) -> bytes:
    """Build one RFC822 message with every header the connector reads."""
    msg = EmailMessage()
    if message_id is not None:
        msg["Message-ID"] = message_id
    msg["Subject"] = subject
    msg["From"] = "sender@example.test"
    msg["To"] = to
    if cc is not None:
        msg["Cc"] = cc
    if date_header is not None:
        msg["Date"] = date_header
    msg.set_content(body)
    return msg.as_bytes()


def _received_plist(epoch: int) -> bytes:
    """Build a trailer carrying Mail.app's arrival time for the message."""
    return plistlib.dumps({"flags": 1, "date-received": epoch})


def _index_row(
    root: Path,
    row_id: int,
    *,
    mailbox_url: str | None = None,
    version: str = "V10",
) -> None:
    """Add one message row, in mailbox 1 unless a new mailbox URL is given."""
    db = root / version / "MailData" / "Envelope Index"
    with sqlite3.connect(db) as conn:
        mailbox_id = 1
        if mailbox_url is not None:
            highest = conn.execute(
                "SELECT COALESCE(MAX(ROWID), 0) FROM mailboxes"
            ).fetchone()[0]
            mailbox_id = highest + 1
            conn.execute(
                "INSERT INTO mailboxes VALUES (?, ?)", (mailbox_id, mailbox_url)
            )
        conn.execute("INSERT INTO messages VALUES (?, ?, 0, 2)", (row_id, mailbox_id))


def _write_emlx(
    root: Path,
    account_id: str,
    filename: str,
    raw: bytes,
    *,
    trailer: bytes = _TRAILING_PLIST,
    mailbox_dir: str = "Inbox.mbox",
    version: str = "V10",
) -> None:
    """Write one ``.emlx`` file, byte count and trailer included."""
    messages = root / version / account_id / mailbox_dir / "Data" / "Messages"
    messages.mkdir(parents=True, exist_ok=True)
    (messages / filename).write_bytes(str(len(raw)).encode() + b"\n" + raw + trailer)


def _by_title(docs) -> dict[str, object]:
    return {doc.title: doc for doc in docs}


def test_sync_is_account_scoped_and_parses_local_message(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    connector = AppleMailConnector(str(config), str(root))

    docs = list(connector.sync())

    assert connector.is_connected()
    assert len(docs) == 1
    assert docs[0].doc_id == "apple_mail:ACCOUNT-ID:<expected@example.test>"
    assert docs[0].title == "Expected subject"
    assert docs[0].content.strip() == "Expected body"
    assert docs[0].participants == ["jfaber@caltech.edu"]
    assert docs[0].metadata["account_id"] == "ACCOUNT-ID"


def test_emlx_body_stops_at_the_declared_byte_count(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    connector = AppleMailConnector(str(config), str(root))

    body = list(connector.sync())[0].content

    assert "plist" not in body
    assert body.strip() == "Expected body"


def test_emlx_without_a_byte_count_is_skipped(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    message = (
        root / "V10" / "ACCOUNT-ID" / "Inbox.mbox" / "Data" / "Messages" / "42.emlx"
    )
    message.write_bytes(_message())

    assert list(AppleMailConnector(str(config), str(root)).sync()) == []


@pytest.mark.parametrize("scheme", ["imap", "pop", "local", "ews"])
def test_sync_covers_every_account_protocol(tmp_path: Path, scheme: str) -> None:
    root, config = _fixture(tmp_path, scheme=scheme)

    docs = list(AppleMailConnector(str(config), str(root)).sync())

    assert [doc.metadata["account_id"] for doc in docs] == ["ACCOUNT-ID"]


def test_available_accounts_reports_protocol_without_reading_messages(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path, scheme="imap", write_config=False)

    accounts = AppleMailConnector(str(config), str(root)).available_accounts()

    assert accounts == [
        {
            "account_id": "ACCOUNT-ID",
            "protocol": "imap",
            "mail_version": "V10",
        }
    ]


def test_configure_writes_the_selected_account(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, scheme="imap", write_config=False)
    connector = AppleMailConnector(str(config), str(root))
    assert not connector.is_connected()

    connector.configure("ACCOUNT-ID")

    assert json.loads(config.read_text()) == {
        "account_id": "ACCOUNT-ID",
        "protocol": "imap",
    }
    assert connector.is_connected()
    assert len(list(connector.sync())) == 1


def test_configure_rejects_an_account_that_is_not_present_locally(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path, write_config=False)
    connector = AppleMailConnector(str(config), str(root))

    with pytest.raises(ValueError, match="ACCOUNT-ID"):
        connector.configure("MISSING-ACCOUNT")

    assert not config.exists()
    assert not connector.is_connected()


def test_configure_requires_a_non_empty_account_id(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, write_config=False)

    with pytest.raises(ValueError, match="required"):
        AppleMailConnector(str(config), str(root)).configure("  ")


def test_configure_is_idempotent_for_the_account_already_pinned(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path, scheme="imap")
    connector = AppleMailConnector(str(config), str(root))

    connector.configure("ACCOUNT-ID")

    assert json.loads(config.read_text())["account_id"] == "ACCOUNT-ID"


def test_configure_refuses_to_repin_a_connected_connector(tmp_path: Path) -> None:
    """Switching accounts must go through disconnect, which purges and resets.

    Overwriting the id in place would leave the first account's mail indexed
    under this connector and start the second account at the first one's
    watermark.
    """
    root, config = _fixture(tmp_path, scheme="imap", account_id="ACCOUNT-A")
    _add_account(root, "ACCOUNT-B", scheme="imap")
    connector = AppleMailConnector(str(config), str(root))

    with pytest.raises(ValueError, match="Disconnect it first"):
        connector.configure("ACCOUNT-B")

    assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"
    assert [doc.metadata["account_id"] for doc in connector.sync()] == ["ACCOUNT-A"]


def test_account_switch_after_disconnect_indexes_only_the_new_account(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path, scheme="imap", account_id="ACCOUNT-A")
    _add_account(root, "ACCOUNT-B", scheme="imap")
    connector = AppleMailConnector(str(config), str(root))

    connector.disconnect()
    connector.configure("ACCOUNT-B")

    assert [doc.metadata["account_id"] for doc in connector.sync()] == ["ACCOUNT-B"]


def test_an_unreadable_index_raises_instead_of_reporting_no_accounts(
    tmp_path: Path,
) -> None:
    """A missing Full Disk Access grant must not read as "no mail accounts"."""
    root, config = _fixture(tmp_path, write_config=False)
    _corrupt_index(root)

    with pytest.raises(_store_error(), match="Envelope Index"):
        AppleMailConnector(str(config), str(root)).available_accounts()


def test_an_unreadable_index_fails_the_sync_instead_of_completing_it(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path)
    _corrupt_index(root)

    with pytest.raises(_store_error(), match="Envelope Index"):
        list(AppleMailConnector(str(config), str(root)).sync())


def test_a_failed_sync_records_the_error_without_advancing_the_watermark(
    tmp_path: Path,
) -> None:
    """The failure must not look like a completed empty sync.

    A generator that merely returns ends normally, so SyncEngine would clear
    the error and move ``last_sync`` past mail that was never read.
    """
    root, config = _fixture(tmp_path)
    connector = AppleMailConnector(str(config), str(root))
    state_db = str(tmp_path / "sync_state.db")

    with KnowledgeStore(str(tmp_path / "knowledge.db")) as store:
        with SyncEngine(
            pipeline=IngestionPipeline(store=store), state_db=state_db
        ) as engine:
            assert engine.sync(connector) == 1
            completed = engine.get_checkpoint("apple_mail")
            assert completed is not None and completed["last_sync"]

            _corrupt_index(root)
            with pytest.raises(_store_error()):
                engine.sync(connector)

            failed = engine.get_checkpoint("apple_mail")

    assert failed is not None
    assert failed["error"]
    assert failed["last_sync"] == completed["last_sync"]


@pytest.mark.parametrize(
    "date_header",
    [
        None,  # no Date header at all
        "not a date at all",
        "Mon, 01 Sep 2025 12:00:00 -0000",  # RFC 5322 "zone unknown"
    ],
)
def test_a_message_without_a_usable_date_still_syncs(
    tmp_path: Path, date_header: str | None
) -> None:
    """``since`` is aware, so a naive timestamp would raise on comparison."""
    root, config = _fixture(tmp_path)
    _write_message(root, "ACCOUNT-ID", "42", date_header=date_header)
    connector = AppleMailConnector(str(config), str(root))

    docs = list(connector.sync(since=datetime(2000, 1, 1, tzinfo=timezone.utc)))

    assert len(docs) == 1
    assert docs[0].timestamp.tzinfo is not None


def test_a_newer_store_wins_over_a_lexicographically_larger_one(
    tmp_path: Path,
) -> None:
    """``V9`` sorts above ``V10`` as text; ``V10`` is the newer store."""
    root, config = _fixture(tmp_path, scheme="imap")
    _add_account(root, "ACCOUNT-ID", scheme="imap", version="V9", subject="Superseded")

    accounts = AppleMailConnector(str(config), str(root)).available_accounts()
    docs = list(AppleMailConnector(str(config), str(root)).sync())

    assert [account["mail_version"] for account in accounts] == ["V10"]
    assert [doc.title for doc in docs] == ["Expected subject"]


def test_a_partially_downloaded_message_is_indexed_and_flagged(
    tmp_path: Path,
) -> None:
    """``<row>.partial.emlx`` holds mail Mail.app has not finished fetching.

    Its headers and whatever body arrived are already readable, so skipping
    the file would hide a message the user can see in Mail. It is indexed
    like any other and the document records which one it was.
    """
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.partial.emlx",
        _raw_message(subject="Still downloading", message_id="<partial@example.test>"),
    )

    docs = _by_title(AppleMailConnector(str(config), str(root)).sync())

    assert sorted(docs) == ["Expected subject", "Still downloading"]
    assert docs["Still downloading"].metadata["partial"] is True
    assert docs["Expected subject"].metadata["partial"] is False


def test_the_complete_copy_wins_when_both_files_exist_for_one_row(
    tmp_path: Path,
) -> None:
    """Mail.app can leave the partial file behind after the full download."""
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.partial.emlx",
        _raw_message(subject="Partial copy", message_id="<partial@example.test>"),
    )
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(subject="Complete copy", message_id="<complete@example.test>"),
    )

    docs = _by_title(AppleMailConnector(str(config), str(root)).sync())

    assert sorted(docs) == ["Complete copy", "Expected subject"]
    assert docs["Complete copy"].metadata["partial"] is False


def test_a_message_without_a_message_id_is_identified_by_its_row(
    tmp_path: Path,
) -> None:
    """A local row id is stable within the store, so it stands in."""
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(subject="Unidentified", message_id=None),
    )

    doc = _by_title(AppleMailConnector(str(config), str(root)).sync())["Unidentified"]

    assert doc.doc_id == "apple_mail:ACCOUNT-ID:43"
    assert doc.metadata["message_id"] == ""
    assert doc.source_id == "ACCOUNT-ID:Inbox:43"


def test_incremental_sync_filters_on_the_trailer_receipt_date(
    tmp_path: Path,
) -> None:
    """``date-received`` is when this store took delivery of the message.

    That is what "everything since my last sync" asks about, so it decides
    the cutoff in both directions: an old message that arrived recently is
    kept, and a recently dated message that arrived long ago is not.
    """
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(
            subject="Old header, new arrival",
            date_header=_OLD_DATE,
            message_id="<arrived-late@example.test>",
        ),
        trailer=_received_plist(_RECEIVED_EPOCH),
    )
    _index_row(root, 44)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "44.emlx",
        _raw_message(
            subject="New header, old arrival",
            message_id="<arrived-early@example.test>",
        ),
        trailer=_received_plist(_OLD_EPOCH),
    )

    docs = _by_title(AppleMailConnector(str(config), str(root)).sync(since=_CUTOFF))

    assert "Old header, new arrival" in docs
    assert "New header, old arrival" not in docs
    assert docs["Old header, new arrival"].timestamp == _RECEIVED_AT


def test_a_message_without_a_trailer_date_is_filtered_on_its_header_date(
    tmp_path: Path,
) -> None:
    """With nothing from Mail.app to go on, the sender's ``Date`` decides."""
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(
            subject="Header only",
            date_header=_OLD_DATE,
            message_id="<header-only@example.test>",
        ),
    )
    connector = AppleMailConnector(str(config), str(root))

    assert "Header only" not in _by_title(connector.sync(since=_CUTOFF))

    kept = _by_title(connector.sync(since=datetime(2019, 1, 1, tzinfo=timezone.utc)))
    assert kept["Header only"].timestamp == datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_the_mailbox_name_comes_from_the_index_url(tmp_path: Path) -> None:
    """The index stores a URL; a reader wants the mailbox it stands for."""
    root, config = _fixture(tmp_path)
    _index_row(root, 43, mailbox_url="imap://ACCOUNT-ID/Archive/Saved%20Mail")
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(subject="Filed", message_id="<filed@example.test>"),
        mailbox_dir="Archive.mbox",
    )

    docs = _by_title(AppleMailConnector(str(config), str(root)).sync())

    assert docs["Filed"].metadata["mailbox"] == "Archive/Saved Mail"
    assert docs["Filed"].channel == "Archive/Saved Mail"
    assert docs["Expected subject"].channel == "Inbox"


def test_each_recipient_is_reported_as_its_own_address(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(
            subject="Addressed",
            message_id="<addressed@example.test>",
            to="one@example.test, two@example.test",
            cc="three@example.test",
        ),
    )

    doc = _by_title(AppleMailConnector(str(config), str(root)).sync())["Addressed"]

    assert doc.participants == [
        "one@example.test",
        "two@example.test",
        "three@example.test",
    ]
    assert doc.participants_raw == doc.participants
    assert doc.source_id == "<addressed@example.test>"
    assert doc.metadata["emlx_uid"] == "43"
    assert doc.metadata["message_id"] == "<addressed@example.test>"


def test_a_comma_inside_a_quoted_name_does_not_invent_a_recipient(
    tmp_path: Path,
) -> None:
    """Three mailboxes are three recipients, whatever their names contain."""
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(
            subject="Quoted",
            message_id="<quoted@example.test>",
            to='"Doe, Jane" <jane@example.test>, John <john@example.test>',
            cc='"Roe, Sam" <sam@example.test>',
        ),
    )

    doc = _by_title(AppleMailConnector(str(config), str(root)).sync())["Quoted"]

    assert doc.participants == [
        '"Doe, Jane" <jane@example.test>',
        "John <john@example.test>",
        '"Roe, Sam" <sam@example.test>',
    ]
    assert doc.participants_raw == doc.participants


def test_an_encoded_recipient_name_arrives_readable(tmp_path: Path) -> None:
    """A reader wants the name, not the encoding that carried it."""
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(
            subject="Encoded",
            message_id="<encoded@example.test>",
            to="=?utf-8?q?Jos=C3=A9_Garc=C3=ADa?= <jose@example.test>",
        ),
    )

    doc = _by_title(AppleMailConnector(str(config), str(root)).sync())["Encoded"]

    assert doc.participants == ["José García <jose@example.test>"]


@pytest.mark.parametrize(
    "trailer",
    [
        b"",  # no trailer at all
        b"not a property list",
        plistlib.dumps(["an array, not a dictionary"]),
    ],
)
def test_an_unusable_trailer_keeps_the_message(tmp_path: Path, trailer: bytes) -> None:
    """Losing Mail.app's metadata is a poor reason to lose the mail."""
    root, config = _fixture(tmp_path)
    _index_row(root, 43)
    _write_emlx(
        root,
        "ACCOUNT-ID",
        "43.emlx",
        _raw_message(
            subject="Bare",
            date_header=_OLD_DATE,
            message_id="<bare@example.test>",
        ),
        trailer=trailer,
    )

    doc = _by_title(AppleMailConnector(str(config), str(root)).sync())["Bare"]

    assert doc.content.strip() == "Expected body"
    assert doc.timestamp == datetime(2020, 1, 1, tzinfo=timezone.utc)
