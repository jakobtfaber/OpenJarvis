"""Tests for ``jarvis connect`` CLI command."""

from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import cli


def test_connect_list_no_connectors() -> None:
    """--list with an empty registry shows a 'no connectors' message."""
    runner = CliRunner()
    with mock.patch(
        "openjarvis.cli.connect_cmd.connect.__wrapped__"
        if hasattr(cli, "__wrapped__")
        else "openjarvis.core.registry.ConnectorRegistry.items",
        return_value=(),
    ):
        with mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.items",
            return_value=(),
        ):
            result = runner.invoke(cli, ["connect", "--list"])

    assert result.exit_code == 0
    assert "No connectors registered" in result.output


def test_connect_list_with_connector(tmp_path: object) -> None:
    """--list with a connector registered shows it in the table."""
    runner = CliRunner()

    # Build a minimal mock connector class
    mock_cls = mock.MagicMock()
    mock_cls.auth_type = "filesystem"
    mock_instance = mock.MagicMock()
    mock_instance.is_connected.return_value = True
    mock_cls.return_value = mock_instance

    with mock.patch(
        "openjarvis.core.registry.ConnectorRegistry.items",
        return_value=(("obsidian", mock_cls),),
    ):
        result = runner.invoke(cli, ["connect", "--list"])

    assert result.exit_code == 0
    assert "obsidian" in result.output


def test_connect_help() -> None:
    """--help exits 0 and mentions the word 'connect'."""
    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "--help"])
    assert result.exit_code == 0
    assert "connect" in result.output.lower()


def test_connect_specific_source(tmp_path: object) -> None:
    """connect --path /nonexistent obsidian shows an error gracefully."""
    runner = CliRunner()

    mock_cls = mock.MagicMock()
    mock_cls.auth_type = "filesystem"
    mock_instance = mock.MagicMock()
    # Path does not exist -> is_connected returns False
    mock_instance.is_connected.return_value = False
    mock_cls.return_value = mock_instance

    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains",
            return_value=True,
        ),
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.get",
            return_value=mock_cls,
        ),
    ):
        # --path before the positional source arg (standard Click group behaviour)
        result = runner.invoke(cli, ["connect", "--path", "/nonexistent", "obsidian"])

    assert result.exit_code == 0
    # Should mention the source and give an indication something went wrong
    assert "obsidian" in result.output or "nonexistent" in result.output


def _apple_mail_cls(accounts: list[dict[str, str]]) -> mock.MagicMock:
    connector_cls = mock.MagicMock()
    connector_cls.auth_type = "local"
    instance = mock.MagicMock()
    instance.available_accounts.return_value = accounts
    instance.is_connected.return_value = True
    connector_cls.return_value = instance
    return connector_cls


def _invoke_apple_mail(connector_cls: mock.MagicMock, *args: str):
    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains", return_value=True
        ),
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.get", return_value=connector_cls
        ),
    ):
        return CliRunner().invoke(cli, ["connect", *args, "apple_mail"])


def test_connect_apple_mail_lists_local_accounts_without_an_account_flag() -> None:
    """Omitting --account must show the choices instead of guessing one."""
    connector_cls = _apple_mail_cls(
        [{"account_id": "ACCOUNT-ID", "protocol": "imap", "mail_version": "V10"}]
    )

    result = _invoke_apple_mail(connector_cls)

    assert result.exit_code == 0
    assert "ACCOUNT-ID" in result.output
    assert "imap" in result.output
    assert "--account" in result.output
    connector_cls.return_value.configure.assert_not_called()


def test_connect_apple_mail_pins_the_requested_account() -> None:
    connector_cls = _apple_mail_cls(
        [{"account_id": "ACCOUNT-ID", "protocol": "imap", "mail_version": "V10"}]
    )

    result = _invoke_apple_mail(connector_cls, "--account", "ACCOUNT-ID")

    assert result.exit_code == 0
    connector_cls.return_value.configure.assert_called_once_with("ACCOUNT-ID")
    assert "connected" in result.output


def test_connect_apple_mail_reports_a_rejected_account() -> None:
    connector_cls = _apple_mail_cls(
        [{"account_id": "ACCOUNT-ID", "protocol": "imap", "mail_version": "V10"}]
    )
    connector_cls.return_value.configure.side_effect = ValueError(
        "Unknown Apple Mail account"
    )

    result = _invoke_apple_mail(connector_cls, "--account", "MISSING")

    assert result.exit_code == 0
    assert "Unknown Apple Mail account" in result.output


def test_connect_apple_mail_without_a_local_store_explains_the_fix() -> None:
    result = _invoke_apple_mail(_apple_mail_cls([]))

    assert result.exit_code == 0
    # Rich wraps at the console width, so compare on the unwrapped text.
    assert "Full Disk Access" in " ".join(result.output.split())


def test_connect_apple_mail_accepts_the_documented_option_order() -> None:
    """``connect`` is a group, so ``--account`` after the source exits 2.

    The setup guide publishes ``jarvis connect --account <ID> apple_mail``;
    this pins that order so the docs cannot drift back to the broken one.
    """
    connector_cls = _apple_mail_cls(
        [{"account_id": "ACCOUNT-ID", "protocol": "imap", "mail_version": "V10"}]
    )

    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains", return_value=True
        ),
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.get", return_value=connector_cls
        ),
    ):
        documented = CliRunner().invoke(
            cli, ["connect", "--account", "ACCOUNT-ID", "apple_mail"]
        )
        reversed_order = CliRunner().invoke(
            cli, ["connect", "apple_mail", "--account", "ACCOUNT-ID"]
        )

    assert documented.exit_code == 0
    connector_cls.return_value.configure.assert_called_once_with("ACCOUNT-ID")
    assert reversed_order.exit_code == 2


class _Response:
    """Just enough of an ``httpx.Response`` for the disconnect handler."""

    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if self._payload is _NO_JSON:
            raise ValueError("not json")
        return self._payload


_NO_JSON = object()


def _disconnect(
    source: str,
    *,
    response: _Response | None = None,
    post_error: Exception | None = None,
    contains: bool = True,
) -> tuple[object, mock.MagicMock, mock.MagicMock, mock.MagicMock]:
    """Run ``--disconnect`` against a stubbed server.

    ``KnowledgeStore`` and ``SyncEngine`` are patched where they are defined so
    the tests can assert the CLI never opens them: reaching them at all would
    mean it had written to a database whose sync worker it does not own. A real
    ``KnowledgeStore()`` would also open the developer's own
    ``~/.openjarvis/knowledge.db``.
    """
    post = mock.MagicMock(
        side_effect=post_error,
        return_value=response
        or _Response(
            200,
            {
                "status": "disconnected",
                "connected": False,
                "purged_sources": [source],
                "retained_sources": [],
            },
        ),
    )
    store_cls = mock.MagicMock()
    engine_cls = mock.MagicMock()

    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains", return_value=contains
        ),
        mock.patch(
            "openjarvis.cli.connect_cmd._server_base_url",
            return_value="http://127.0.0.1:8000",
        ),
        mock.patch("httpx.post", post),
        mock.patch("openjarvis.connectors.store.KnowledgeStore", store_cls),
        mock.patch("openjarvis.connectors.sync_engine.SyncEngine", engine_cls),
    ):
        result = CliRunner().invoke(cli, ["connect", "--disconnect", source])

    return result, post, store_cls, engine_cls


def _flat(result: object) -> str:
    """Rich wraps at the console width; compare on the unwrapped text."""
    return " ".join(result.output.split())  # type: ignore[attr-defined]


def test_connect_disconnect() -> None:
    """--disconnect gmail exits 0."""
    result, post, _, _ = _disconnect("gmail")

    assert result.exit_code == 0
    post.assert_called_once()
    assert "purged" in _flat(result)


def test_disconnect_delegates_to_the_server_that_owns_the_sync_worker() -> None:
    """The CLI has no sync thread; the running service does.

    Purging from here races that worker: a sync paused mid-document resumes
    after the delete, re-ingests the account just disconnected and rewrites its
    checkpoint, so the next account inherits a watermark past its own mail.
    Only the server can cancel the worker and wait for it.
    """
    _, post, store_cls, engine_cls = _disconnect("apple_mail")

    assert post.call_args.args[0] == (
        "http://127.0.0.1:8000/v1/connectors/apple_mail/disconnect"
    )
    store_cls.assert_not_called()
    engine_cls.assert_not_called()


def test_disconnect_refuses_when_the_server_is_unreachable() -> None:
    """Without the server there is nothing that can establish ownership.

    A refusal can be retried; a half-applied local purge cannot be undone.
    A refused connection is also the one transport failure where preservation
    is a fact: the request never left, so nothing on the server ran.
    """
    import httpx

    result, _, store_cls, engine_cls = _disconnect(
        "apple_mail", post_error=httpx.ConnectError("connection refused")
    )

    assert result.exit_code == 1
    output = _flat(result)
    assert "Could not reach" in output
    assert "never sent" in output
    assert "nothing was deleted" in output
    assert "Disconnected apple_mail" not in output
    assert "purged" not in output
    store_cls.assert_not_called()
    engine_cls.assert_not_called()


def test_disconnect_makes_no_claim_about_a_binding_it_never_looked_at() -> None:
    """A refused connection says nothing about the account binding.

    All it proves is that this attempt never reached the server. The source may
    have been bound to an account, or to none at all: no config file is read
    and no credential is opened on this path, so stating that it "is still
    connected" would be an assertion about state the CLI never looked at. The
    absent-binding case is the one that exposes it, and reaching for the state
    to check it would itself be the local access this command must not perform.
    """
    import httpx

    for error in (
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("timed out"),
    ):
        result, _, store_cls, engine_cls = _disconnect("apple_mail", post_error=error)

        assert result.exit_code == 1
        output = _flat(result)
        assert "changed nothing" in output
        assert "says nothing about whether apple_mail is bound" in output
        assert "is still connected" not in output
        assert "stays connected" not in output
        assert "still bound" not in output
        store_cls.assert_not_called()
        engine_cls.assert_not_called()


def test_disconnect_reports_a_lost_response_as_an_unknown_outcome() -> None:
    """A dropped answer is not a dropped request.

    A read timeout or a severed connection happens after the request has gone
    out, so the purge may have run in full. Claiming the account survived would
    be a preservation claim the CLI cannot support.
    """
    import httpx

    for error in (
        httpx.ReadTimeout("timed out waiting for the response"),
        httpx.RemoteProtocolError("server disconnected without sending a response"),
        httpx.ReadError("connection reset"),
    ):
        result, _, store_cls, engine_cls = _disconnect("apple_mail", post_error=error)

        assert result.exit_code == 1
        output = _flat(result)
        assert "may already have been carried out" in output
        assert "is unknown" in output
        assert "Could not reach" not in output
        assert "nothing was deleted" not in output
        assert "left in place" not in output
        assert "Disconnected apple_mail" not in output
        assert "purged" not in output
        store_cls.assert_not_called()
        engine_cls.assert_not_called()


def test_disconnect_reports_a_sync_that_is_still_stopping() -> None:
    """409 means the old worker still owns the store, so nothing was purged."""
    result, _, store_cls, _ = _disconnect(
        "apple_mail",
        response=_Response(
            409,
            {
                "detail": "Sync for 'apple_mail' is still stopping; "
                "indexed content was not purged"
            },
        ),
    )

    assert result.exit_code == 1
    output = _flat(result)
    assert "still stopping" in output
    assert "Disconnected apple_mail" not in output
    store_cls.assert_not_called()


def test_disconnect_reports_a_failed_cleanup_as_still_connected() -> None:
    """A failed purge keeps the account bound, which is what blocks a repin.

    Reporting success here would tell the user they may connect a different
    account on top of the old account's surviving mail. The server labels this
    case, so the claim rests on the label rather than on the wording.
    """
    result, _, store_cls, _ = _disconnect(
        "apple_mail",
        response=_Response(
            500,
            {
                "code": "cleanup_failed",
                "connected": True,
                "detail": (
                    "Indexed-data cleanup failed, still connected: database is locked"
                ),
            },
        ),
    )

    assert result.exit_code == 1
    output = _flat(result)
    assert "database is locked" in output
    assert "still connected" in output
    assert "left in place" in output
    assert "Disconnected apple_mail" not in output
    assert "purged" not in output
    store_cls.assert_not_called()


def test_disconnect_reports_a_failed_revoke_without_claiming_the_content_survived() -> (
    None
):
    """Past the purge, the content is gone even though the account is bound.

    This is the fail-closed shortfall the server accepts on purpose. Saying the
    indexed content was left in place would be false.
    """
    result, _, store_cls, _ = _disconnect(
        "apple_mail",
        response=_Response(
            500,
            {
                "code": "revoke_failed",
                "connected": True,
                "purged_sources": ["apple_mail"],
                "retained_sources": [],
                "detail": (
                    "Indexed content was purged but the account binding could "
                    "not be released: permission denied"
                ),
            },
        ),
    )

    assert result.exit_code == 1
    output = _flat(result)
    assert "permission denied" in output
    assert "indexed content is gone" in output
    assert "still bound" in output
    assert "left in place" not in output
    assert "still connected. Its indexed content" not in output
    assert "Disconnected apple_mail" not in output
    store_cls.assert_not_called()


def test_disconnect_names_the_sources_a_shared_owner_kept_indexed() -> None:
    """A retained source is still indexed, so success cannot claim a purge.

    Gmail's OAuth and IMAP connectors both write ``source='gmail'``. The server
    retains that source while the other one is still connected, which is the
    right call and also means most of what this connector indexed is still
    searchable. "purged its indexed content" would be false for exactly those
    rows.
    """
    result, _, store_cls, _ = _disconnect(
        "gmail",
        response=_Response(
            200,
            {
                "status": "disconnected",
                "connected": False,
                "purged_sources": [],
                "retained_sources": ["gmail"],
            },
        ),
    )

    assert result.exit_code == 0
    output = _flat(result)
    assert "Disconnected gmail." in output
    assert "gmail was kept because ownership may be shared" in output
    assert "purged its indexed content" not in output
    # The field says a source stayed, not that a peer answered a probe. The
    # server retains on an unreadable probe as well, so a connected owner is
    # not a fact this output may state.
    assert "connected integration" not in output
    store_cls.assert_not_called()


def test_disconnect_says_the_purge_scope_is_unknown_when_the_body_omits_it() -> None:
    """A body without the field is unknown scope, never an empty retention set.

    The lifecycle finished, so this is a success. Reading the missing field as
    "nothing was retained" is how a partial purge gets reported as a complete
    one, so the absent case has to say it does not know.
    """
    result, _, store_cls, _ = _disconnect(
        "gmail",
        response=_Response(200, {"status": "disconnected", "connected": False}),
    )

    assert result.exit_code == 0
    output = _flat(result)
    assert "Disconnected gmail." in output
    assert "did not report which indexed content was removed" in output
    assert "purged its indexed content" not in output
    store_cls.assert_not_called()


def test_disconnect_treats_a_malformed_retention_list_as_unknown() -> None:
    """A field of the wrong shape is no more informative than a missing one."""
    for retained in (["gmail", 7], "gmail", {"gmail": True}, 0):
        result, _, store_cls, _ = _disconnect(
            "gmail",
            response=_Response(
                200,
                {
                    "status": "disconnected",
                    "connected": False,
                    "retained_sources": retained,
                },
            ),
        )

        assert result.exit_code == 0
        output = _flat(result)
        assert "did not report which indexed content was removed" in output
        assert "purged its indexed content" not in output
        store_cls.assert_not_called()


def test_disconnect_qualifies_a_failed_revoke_that_also_retained_a_source() -> None:
    """The worst case has to read correctly: bound account, content still there.

    Two shortfalls land together. The binding survived, and a source another
    connected owner writes stayed indexed. Claiming "the indexed content is
    gone" here would be false twice over.
    """
    result, _, store_cls, _ = _disconnect(
        "gmail",
        response=_Response(
            500,
            {
                "code": "revoke_failed",
                "connected": True,
                "purged_sources": [],
                "retained_sources": ["gmail"],
                "detail": (
                    "Indexed content this connector owned alone was purged, and "
                    "gmail stayed indexed because ownership may be shared with "
                    "another integration, but the account binding could not be "
                    "released: permission denied"
                ),
            },
        ),
    )

    assert result.exit_code == 1
    output = _flat(result)
    assert "permission denied" in output
    assert "gmail was kept because ownership may be shared" in output
    assert "still bound" in output
    assert "The indexed content is gone" not in output
    assert "connected integration" not in output
    store_cls.assert_not_called()


def test_disconnect_never_asserts_a_connected_owner_it_cannot_know_about() -> None:
    """Retention also happens when the peer's connectivity cannot be read.

    The server retains a shared source both when it confirms another owner is
    connected and when that owner's ``is_connected()`` raises. ``retained_sources``
    carries the same value either way, so any message claiming a connected peer
    is a guess the CLI is not entitled to make. These are the two bodies the
    failed-probe server path produces.
    """
    success = _Response(
        200,
        {
            "status": "disconnected",
            "connected": False,
            "purged_sources": [],
            "retained_sources": ["gmail"],
        },
    )
    revoke_failure = _Response(
        500,
        {
            "code": "revoke_failed",
            "connected": True,
            "purged_sources": [],
            "retained_sources": ["gmail"],
            "detail": (
                "Indexed content this connector owned alone was purged, and "
                "gmail stayed indexed because ownership may be shared with "
                "another integration, but the account binding could not be "
                "released: permission denied"
            ),
        },
    )

    for response, expected_exit in ((success, 0), (revoke_failure, 1)):
        result, _, store_cls, _ = _disconnect("gmail", response=response)

        assert result.exit_code == expected_exit
        output = _flat(result)
        assert "gmail was kept because ownership may be shared" in output
        assert "connected integration" not in output
        assert "connected owner" not in output
        store_cls.assert_not_called()


def test_disconnect_does_not_state_a_purge_scope_a_failed_revoke_left_out() -> None:
    """Without the field the failure cannot say how much content survived."""
    result, _, store_cls, _ = _disconnect(
        "apple_mail",
        response=_Response(
            500,
            {
                "code": "revoke_failed",
                "connected": True,
                "detail": "the account binding could not be released: denied",
            },
        ),
    )

    assert result.exit_code == 1
    output = _flat(result)
    assert "did not report what it covered" in output
    assert "still bound" in output
    assert "The indexed content is gone" not in output
    store_cls.assert_not_called()


def test_disconnect_does_not_claim_success_without_a_completed_lifecycle() -> None:
    """Only a coherent 200 means the purge actually ran.

    Everything else here is an unknown outcome: a body that does not agree the
    lifecycle finished, a 200 that is not JSON at all, and an unlabelled server
    error that may have failed before or after the delete.
    """
    for response in (
        _Response(200, {"status": "stopping"}),
        _Response(200, {"status": "disconnected", "connected": True}),
        _Response(200, _NO_JSON, text="<html>proxy</html>"),
        _Response(500, {"detail": "unlabelled failure"}),
        _Response(503, _NO_JSON, text="service unavailable"),
    ):
        result, _, store_cls, _ = _disconnect("apple_mail", response=response)

        assert result.exit_code == 1
        output = _flat(result)
        assert "is unknown" in output
        assert "nothing was deleted" not in output
        assert "left in place" not in output
        assert "Disconnected apple_mail" not in output
        assert "purged" not in output
        store_cls.assert_not_called()


def test_disconnect_requires_the_confirmation_field_to_be_exactly_false() -> None:
    """A missing or wrongly typed ``connected`` has confirmed nothing.

    Reading the field as merely falsy turns a malformed answer into a success:
    a body that leaves it out, or sends ``null``, ``0`` or ``""``, would then
    be indistinguishable from the server stating that the account is no longer
    bound. Only the literal ``false`` is a confirmation, and everything else is
    an unknown outcome.
    """
    for response in (
        _Response(200, {"status": "disconnected"}),
        _Response(200, {"status": "disconnected", "connected": None}),
        _Response(200, {"status": "disconnected", "connected": 0}),
        _Response(200, {"status": "disconnected", "connected": ""}),
        _Response(200, {"status": "disconnected", "connected": "false"}),
        _Response(200, {"connected": False}),
    ):
        result, _, store_cls, _ = _disconnect("apple_mail", response=response)

        assert result.exit_code == 1
        output = _flat(result)
        assert "is unknown" in output
        assert "nothing was deleted" not in output
        assert "left in place" not in output
        assert "Disconnected apple_mail" not in output
        assert "purged" not in output
        store_cls.assert_not_called()


def test_disconnect_retried_after_an_unknown_outcome_can_still_succeed() -> None:
    """The advice after an unknown outcome has to be advice that works.

    A disconnect the server already completed answers 404 on the retry, and a
    disconnect it never reached answers 200. Neither leaves the CLI claiming
    something it cannot see.
    """
    import httpx

    ambiguous, _, _, _ = _disconnect(
        "apple_mail", post_error=httpx.ReadTimeout("no response")
    )
    assert ambiguous.exit_code == 1
    assert "run this again" in _flat(ambiguous).lower()

    completed, _, store_cls, _ = _disconnect("apple_mail")
    assert completed.exit_code == 0
    assert "purged" in _flat(completed)
    store_cls.assert_not_called()

    already_gone, _, _, _ = _disconnect(
        "apple_mail",
        response=_Response(404, {"detail": "Connector 'apple_mail' not found"}),
    )
    assert already_gone.exit_code == 1
    assert "is unknown" in _flat(already_gone)


def test_disconnect_unknown_source_never_reaches_the_server() -> None:
    result, post, _, _ = _disconnect("nope", contains=False)

    assert result.exit_code == 1
    assert "Unknown source" in result.output
    post.assert_not_called()


def test_server_base_url_dials_a_wildcard_bind_address_on_the_loopback() -> None:
    """``0.0.0.0`` is where the server listens, not somewhere to send to."""
    from openjarvis.cli import connect_cmd

    config = mock.MagicMock()
    config.server.host = "0.0.0.0"
    config.server.port = 9100

    with mock.patch("openjarvis.core.config.load_config", return_value=config):
        assert connect_cmd._server_base_url() == "http://127.0.0.1:9100"

    config.server.host = "127.0.0.1"
    with mock.patch("openjarvis.core.config.load_config", return_value=config):
        assert connect_cmd._server_base_url() == "http://127.0.0.1:9100"
