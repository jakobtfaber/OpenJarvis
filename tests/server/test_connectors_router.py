"""Tests for the /v1/connectors API router."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def app():
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    from openjarvis.server.connectors_router import create_connectors_router

    _app = FastAPI()
    router = create_connectors_router()
    # The router already carries ``prefix="/v1/connectors"``. The real
    # ``server/app.py`` mounts it with ``include_router(router)`` — adding
    # a second ``prefix="/v1"`` here would produce ``/v1/v1/connectors``
    # and every request would 404.
    _app.include_router(router)
    return TestClient(_app)


def test_list_connectors(app):
    """GET /v1/connectors returns a list that includes the obsidian connector."""
    resp = app.get("/v1/connectors")
    assert resp.status_code == 200
    data = resp.json()
    assert "connectors" in data
    ids = [c["connector_id"] for c in data["connectors"]]
    assert "obsidian" in ids


def test_connector_detail(app):
    """GET /v1/connectors/obsidian returns the expected fields."""
    resp = app.get("/v1/connectors/obsidian")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connector_id"] == "obsidian"
    assert "display_name" in data
    assert "auth_type" in data
    assert "connected" in data
    assert "mcp_tools" in data


def test_connector_not_found(app):
    """GET /v1/connectors/nonexistent returns 404."""
    resp = app.get("/v1/connectors/nonexistent")
    assert resp.status_code == 404


def test_connect_obsidian(app, tmp_path):
    """POST /v1/connectors/obsidian/connect with a valid path marks it connected."""
    # Create a minimal vault directory so is_connected() returns True.
    vault = tmp_path / "vault"
    vault.mkdir()

    resp = app.post("/v1/connectors/obsidian/connect", json={"path": str(vault)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["connector_id"] == "obsidian"
    assert data["connected"] is True


def test_disconnect(app):
    """POST /v1/connectors/obsidian/disconnect returns 200 with connected=False."""
    resp = app.post("/v1/connectors/obsidian/disconnect")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connector_id"] == "obsidian"
    assert data["connected"] is False


def test_disconnect_clears_stale_sync_checkpoint(app) -> None:
    """Disconnecting a connector must reset its sync checkpoint.

    Regression: a user disconnects Obsidian from one vault and reconnects
    it to a different vault. Before this fix, the SyncEngine checkpoint
    (items_synced, cursor, last_sync) was keyed only by connector_id and
    survived disconnect, so the next sync resumed from the OLD vault's
    watermark -- inflating the reported item count with stale data and
    risking skipped items in the new vault whose timestamps predate the
    old watermark.
    """
    from openjarvis.connectors.pipeline import IngestionPipeline
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.connectors.sync_engine import SyncEngine

    with KnowledgeStore() as store:
        with SyncEngine(pipeline=IngestionPipeline(store=store)) as engine:
            engine._save_checkpoint("obsidian", 9, cursor="stale-cursor")
            assert engine.get_checkpoint("obsidian") is not None

    resp = app.post("/v1/connectors/obsidian/disconnect")
    assert resp.status_code == 200

    # A fresh SyncEngine (same default state DB) must see no checkpoint.
    with KnowledgeStore() as store:
        with SyncEngine(pipeline=IngestionPipeline(store=store)) as fresh_engine:
            assert fresh_engine.get_checkpoint("obsidian") is None


def test_disconnect_purges_previously_ingested_content(app) -> None:
    """Disconnecting a connector must purge its previously-ingested chunks.

    Regression: reconnecting Obsidian to a different vault left the OLD
    vault's chunks in the KnowledgeStore forever (disconnect only cleared
    credentials, never indexed content). If the new vault contains a file
    at the same relative path as one in the old vault (e.g. both have a
    "Welcome.md"), the resulting doc_id collides with the orphaned old
    chunk, and the ingestion pipeline's duplicate-doc_id dedup silently
    discards the new content -- the user's real notes never get indexed,
    with no error surfaced anywhere.
    """
    from openjarvis.connectors.store import KnowledgeStore

    with KnowledgeStore() as store:
        store.store(
            content="Stale content from a previously-connected vault",
            source="obsidian",
            doc_type="note",
            doc_id="obsidian:Welcome.md",
            title="Welcome",
            author="",
        )
        assert any(
            r.metadata.get("source") == "obsidian"
            for r in store.retrieve("stale content vault", top_k=10)
        )

    resp = app.post("/v1/connectors/obsidian/disconnect")
    assert resp.status_code == 200

    with KnowledgeStore() as fresh_store:
        assert not any(
            r.metadata.get("source") == "obsidian"
            for r in fresh_store.retrieve("stale content vault", top_k=10)
        )


def test_disconnect_cancels_inflight_sync_before_purge(app) -> None:
    from openjarvis.connectors._stubs import Document, SyncStatus
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.server.connectors_router import _instances

    started = threading.Event()
    released = threading.Event()
    disconnect_called = threading.Event()

    class BlockingConnector:
        connector_id = "obsidian"
        indexed_sources = ("obsidian",)

        def is_connected(self):
            return True

        def sync(self, **kwargs):
            started.set()
            released.wait(timeout=2)
            yield Document(
                doc_id="obsidian:late-write",
                source="obsidian",
                doc_type="note",
                content="must not survive disconnect",
            )

        def disconnect(self):
            disconnect_called.set()

        def sync_status(self):
            return SyncStatus()

    _instances["obsidian"] = BlockingConnector()
    responses = []

    def disconnect_request():
        responses.append(app.post("/v1/connectors/obsidian/disconnect"))

    disconnect_thread = threading.Thread(target=disconnect_request)
    try:
        assert app.post("/v1/connectors/obsidian/sync").status_code == 200
        assert started.wait(timeout=2)
        disconnect_thread.start()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if app.get("/v1/connectors/obsidian/sync").json()["state"] == "stopping":
                break
            time.sleep(0.01)
        else:
            pytest.fail("disconnect never entered the stopping state")

        # Credentials remain intact until the old writer has stopped.
        assert not disconnect_called.is_set()
        released.set()
        disconnect_thread.join(timeout=3)
        assert not disconnect_thread.is_alive()
        assert responses[0].status_code == 200
        assert disconnect_called.is_set()
        with KnowledgeStore() as store:
            assert not any(
                result.metadata.get("doc_id") == "obsidian:late-write"
                for result in store.retrieve("must survive disconnect", top_k=10)
            )
    finally:
        released.set()
        disconnect_thread.join(timeout=3)
        _instances.pop("obsidian", None)


def test_disconnect_timeout_preserves_source_and_guards_reconnect(
    app,
    monkeypatch,
    tmp_path,
) -> None:
    from openjarvis.connectors._stubs import SyncStatus
    from openjarvis.server import connectors_router
    from openjarvis.server.connectors_router import _instances

    started = threading.Event()
    released = threading.Event()
    finished = threading.Event()
    disconnect_called = threading.Event()

    class StuckConnector:
        connector_id = "obsidian"
        indexed_sources = ("obsidian",)
        auth_type = "filesystem"

        def __init__(self):
            self._connected = True
            self._vault_path = "old-vault"

        def is_connected(self):
            return self._connected

        def sync(self, **kwargs):
            started.set()
            released.wait(timeout=3)
            finished.set()
            if False:
                yield

        def disconnect(self):
            disconnect_called.set()
            self._connected = False

        def sync_status(self):
            return SyncStatus()

    connector = StuckConnector()
    _instances["obsidian"] = connector
    monkeypatch.setattr(connectors_router, "_SYNC_STOP_TIMEOUT_SECONDS", 0.05)
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()

    try:
        assert app.post("/v1/connectors/obsidian/sync").status_code == 200
        assert started.wait(timeout=2)

        response = app.post("/v1/connectors/obsidian/disconnect")
        assert response.status_code == 409
        assert connector.is_connected()
        assert connector._vault_path == "old-vault"
        assert not disconnect_called.is_set()

        reconnect = app.post(
            "/v1/connectors/obsidian/connect",
            json={"path": str(new_vault)},
        )
        assert reconnect.status_code == 409
        assert connector._vault_path == "old-vault"
        assert app.post("/v1/connectors/obsidian/sync").status_code == 409

        released.set()
        assert finished.wait(timeout=2)
        assert app.get("/v1/connectors/obsidian/sync").json()["state"] == "stopping"
        response = app.post("/v1/connectors/obsidian/disconnect")
        assert response.status_code == 200
        assert disconnect_called.is_set()
        assert not connector.is_connected()
    finally:
        released.set()
        _instances.pop("obsidian", None)


def test_disconnect_prevents_sync_start_racing_checkpoint_read(
    app,
    monkeypatch,
) -> None:
    """Disconnect waits for an in-progress start decision, then purges it."""
    from openjarvis.connectors._stubs import Document, SyncStatus
    from openjarvis.connectors.sync_engine import SyncEngine
    from openjarvis.server.connectors_router import _instances

    baseline_started = threading.Event()
    release_baseline = threading.Event()
    sync_called = threading.Event()
    gate_lock = threading.Lock()
    gate_first_call = True
    original_get_checkpoint = SyncEngine.get_checkpoint

    def gated_get_checkpoint(self, connector_id):
        nonlocal gate_first_call
        with gate_lock:
            should_gate = gate_first_call
            gate_first_call = False
        if should_gate:
            baseline_started.set()
            assert release_baseline.wait(timeout=3)
        return original_get_checkpoint(self, connector_id)

    monkeypatch.setattr(SyncEngine, "get_checkpoint", gated_get_checkpoint)

    class RacingConnector:
        connector_id = "obsidian"
        indexed_sources = ("obsidian",)

        def __init__(self):
            self.connected = True

        def is_connected(self):
            return self.connected

        def sync(self, **kwargs):
            sync_called.set()
            yield Document(
                doc_id="obsidian:post-disconnect",
                source="obsidian",
                doc_type="note",
                content="must never be written",
            )

        def disconnect(self):
            self.connected = False

        def sync_status(self):
            return SyncStatus()

    _instances["obsidian"] = RacingConnector()
    sync_response = []
    disconnect_response = []
    disconnect_done = threading.Event()

    def trigger_sync():
        sync_response.append(app.post("/v1/connectors/obsidian/sync"))

    def disconnect():
        disconnect_response.append(app.post("/v1/connectors/obsidian/disconnect"))
        disconnect_done.set()

    request_thread = threading.Thread(target=trigger_sync)
    disconnect_thread = threading.Thread(target=disconnect)
    try:
        request_thread.start()
        assert baseline_started.wait(timeout=3)

        disconnect_thread.start()
        # The checkpoint snapshot and thread publication are one lifecycle
        # operation. Disconnect must not overtake it and purge too early.
        assert not disconnect_done.wait(timeout=0.1)

        release_baseline.set()
        request_thread.join(timeout=3)
        disconnect_thread.join(timeout=3)
        assert not request_thread.is_alive()
        assert not disconnect_thread.is_alive()
        assert sync_response[0].status_code == 200
        assert sync_response[0].json()["status"] == "started"
        assert disconnect_response[0].status_code == 200

        from openjarvis.connectors.store import KnowledgeStore

        with KnowledgeStore() as store:
            assert not any(
                result.metadata.get("doc_id") == "obsidian:post-disconnect"
                for result in store.retrieve("must never be written", top_k=10)
            )
    finally:
        release_baseline.set()
        request_thread.join(timeout=3)
        disconnect_thread.join(timeout=3)
        _instances.pop("obsidian", None)


def test_disconnect_preserves_source_owned_by_connected_peer(app) -> None:
    from openjarvis.connectors._stubs import SyncStatus
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.server.connectors_router import _instances

    class FakeConnector:
        def __init__(self, connector_id, indexed_sources=()):
            self.connector_id = connector_id
            self.indexed_sources = indexed_sources
            self.connected = True

        def is_connected(self):
            return self.connected

        def disconnect(self):
            self.connected = False

        def sync_status(self):
            return SyncStatus()

    _instances["gmail"] = FakeConnector("gmail")
    _instances["gmail_imap"] = FakeConnector("gmail_imap", ("gmail",))
    try:
        with KnowledgeStore() as store:
            store.store(
                content="shared gmail ownership sentinel",
                source="gmail",
                doc_type="email",
                doc_id="gmail:shared-owner",
            )

        response = app.post("/v1/connectors/gmail_imap/disconnect")
        assert response.status_code == 200

        with KnowledgeStore() as store:
            assert any(
                result.metadata.get("doc_id") == "gmail:shared-owner"
                for result in store.retrieve("ownership sentinel", top_k=10)
            )
            store.delete_by_source("gmail")
    finally:
        _instances.pop("gmail", None)
        _instances.pop("gmail_imap", None)


class _SharedSourceConnector:
    """A connector that writes a source another connector also owns."""

    def __init__(
        self,
        connector_id,
        indexed_sources=(),
        revoke_error=None,
        probe_error=None,
        connected=True,
    ):
        from openjarvis.connectors._stubs import SyncStatus

        self._status = SyncStatus
        self.connector_id = connector_id
        self.indexed_sources = indexed_sources
        self.revoke_error = revoke_error
        self.probe_error = probe_error
        self.connected = connected

    def is_connected(self):
        if self.probe_error is not None:
            raise self.probe_error
        return self.connected

    def disconnect(self):
        if self.revoke_error is not None:
            raise self.revoke_error
        self.connected = False

    def sync_status(self):
        return self._status()


def test_disconnect_reports_the_source_it_kept_for_a_connected_peer(app) -> None:
    """Retention is right, and the response has to admit it happened.

    Both Gmail connectors write ``source='gmail'``, so disconnecting one leaves
    every row indexed for the other. A body that did not distinguish this from
    a full purge would let the CLI tell the user content is gone while it is
    still searchable.
    """
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.server.connectors_router import _instances

    _instances["gmail"] = _SharedSourceConnector("gmail")
    _instances["gmail_imap"] = _SharedSourceConnector("gmail_imap", ("gmail",))
    try:
        with KnowledgeStore() as store:
            store.store(
                content="shared gmail retention reporting sentinel",
                source="gmail",
                doc_type="email",
                doc_id="gmail:retention-report",
            )

        response = app.post("/v1/connectors/gmail_imap/disconnect")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "disconnected"
        assert body["retained_sources"] == ["gmail"]
        assert body["purged_sources"] == []

        with KnowledgeStore() as store:
            assert any(
                result.metadata.get("doc_id") == "gmail:retention-report"
                for result in store.retrieve("retention reporting sentinel", top_k=10)
            )
            store.delete_by_source("gmail")
    finally:
        _instances.pop("gmail", None)
        _instances.pop("gmail_imap", None)


def test_a_failed_revoke_does_not_claim_a_purge_that_retention_prevented(app) -> None:
    """The two shortfalls can land together, and the detail must carry both.

    The binding survived the failure and the shared source was never in the
    purge set. "Indexed content was purged" is false on both counts, so the
    detail names what stayed instead.
    """
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.server.connectors_router import _instances

    _instances["gmail"] = _SharedSourceConnector("gmail")
    _instances["gmail_imap"] = _SharedSourceConnector(
        "gmail_imap", ("gmail",), revoke_error=OSError("permission denied")
    )
    try:
        with KnowledgeStore() as store:
            store.store(
                content="shared gmail failed revoke sentinel",
                source="gmail",
                doc_type="email",
                doc_id="gmail:failed-revoke",
            )

        response = app.post("/v1/connectors/gmail_imap/disconnect")
        assert response.status_code == 500
        body = response.json()
        assert body["code"] == "revoke_failed"
        assert body["connected"] is True
        assert body["retained_sources"] == ["gmail"]
        assert body["purged_sources"] == []
        detail = body["detail"]
        assert "gmail stayed indexed because ownership may be shared" in detail
        assert "permission denied" in detail
        assert "Indexed content was purged but" not in detail
        assert "connected owner" not in detail

        with KnowledgeStore() as store:
            assert any(
                result.metadata.get("doc_id") == "gmail:failed-revoke"
                for result in store.retrieve("failed revoke sentinel", top_k=10)
            )
            store.delete_by_source("gmail")
    finally:
        _instances.pop("gmail", None)
        _instances.pop("gmail_imap", None)


def test_retention_on_an_unreadable_probe_claims_no_connected_owner(app) -> None:
    """Retaining on a failed probe is right, and the wording has to fit it.

    The ownership filter also retains a shared source when the peer's
    ``is_connected()`` raises, which is the safe call: unreadable ownership is
    not proof that nothing else needs the rows. Here the peer is disconnected
    and its probe raises, so no connected owner exists at all. A response
    asserting one would be false, and the retained rows still have to survive.
    """
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.server.connectors_router import _instances

    _instances["gmail"] = _SharedSourceConnector(
        "gmail", connected=False, probe_error=OSError("keychain unavailable")
    )
    _instances["gmail_imap"] = _SharedSourceConnector("gmail_imap", ("gmail",))
    try:
        with KnowledgeStore() as store:
            store.store(
                content="shared gmail unreadable probe sentinel",
                source="gmail",
                doc_type="email",
                doc_id="gmail:unreadable-probe",
            )

        response = app.post("/v1/connectors/gmail_imap/disconnect")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "disconnected"
        assert body["retained_sources"] == ["gmail"]
        assert body["purged_sources"] == []

        with KnowledgeStore() as store:
            assert any(
                result.metadata.get("doc_id") == "gmail:unreadable-probe"
                for result in store.retrieve("unreadable probe sentinel", top_k=10)
            )
            store.delete_by_source("gmail")
    finally:
        _instances.pop("gmail", None)
        _instances.pop("gmail_imap", None)


def test_a_failed_revoke_after_an_unreadable_probe_states_only_shared_ownership(
    app,
) -> None:
    """Both failures at once, with ownership that was never established.

    The binding survived, the shared source was retained on an unreadable
    probe, and no peer is connected. The detail may say ownership might be
    shared; it may not say another owner is connected.
    """
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.server.connectors_router import _instances

    _instances["gmail"] = _SharedSourceConnector(
        "gmail", connected=False, probe_error=OSError("keychain unavailable")
    )
    _instances["gmail_imap"] = _SharedSourceConnector(
        "gmail_imap", ("gmail",), revoke_error=OSError("permission denied")
    )
    try:
        with KnowledgeStore() as store:
            store.store(
                content="shared gmail unreadable probe revoke sentinel",
                source="gmail",
                doc_type="email",
                doc_id="gmail:unreadable-probe-revoke",
            )

        response = app.post("/v1/connectors/gmail_imap/disconnect")
        assert response.status_code == 500
        body = response.json()
        assert body["code"] == "revoke_failed"
        assert body["connected"] is True
        assert body["retained_sources"] == ["gmail"]
        assert body["purged_sources"] == []
        detail = body["detail"]
        assert "gmail stayed indexed because ownership may be shared" in detail
        assert "permission denied" in detail
        assert "connected owner" not in detail
        assert "Indexed content was purged but" not in detail

        with KnowledgeStore() as store:
            assert any(
                result.metadata.get("doc_id") == "gmail:unreadable-probe-revoke"
                for result in store.retrieve("unreadable probe revoke", top_k=10)
            )
            store.delete_by_source("gmail")
    finally:
        _instances.pop("gmail", None)
        _instances.pop("gmail_imap", None)


def test_disconnect_restores_checkpoint_when_purge_fails(app, monkeypatch) -> None:
    from openjarvis.connectors.pipeline import IngestionPipeline
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.connectors.sync_engine import SyncEngine
    from openjarvis.server.connectors_router import _instances

    class FakeObsidian:
        indexed_sources = ("obsidian",)

        def is_connected(self):
            return True

        def disconnect(self):
            pass

    with KnowledgeStore() as store:
        with SyncEngine(pipeline=IngestionPipeline(store=store)) as engine:
            engine._save_checkpoint("obsidian", 17, cursor="keep-me")

    def fail_purge(self, sources):
        raise RuntimeError("simulated purge failure")

    monkeypatch.setattr(KnowledgeStore, "delete_by_sources", fail_purge)
    _instances["obsidian"] = FakeObsidian()
    try:
        response = app.post("/v1/connectors/obsidian/disconnect")
        assert response.status_code == 500
        with KnowledgeStore() as store:
            with SyncEngine(pipeline=IngestionPipeline(store=store)) as engine:
                checkpoint = engine.get_checkpoint("obsidian")
                assert checkpoint is not None
                assert checkpoint["items_synced"] == 17
                assert checkpoint["cursor"] == "keep-me"
                engine.reset_checkpoint("obsidian")
    finally:
        _instances.pop("obsidian", None)


def test_sync_status(app):
    """GET /v1/connectors/obsidian/sync returns a response with a state field."""
    resp = app.get("/v1/connectors/obsidian/sync")
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert data["connector_id"] == "obsidian"


def test_trigger_sync(app, tmp_path: Path) -> None:
    """POST /v1/connectors/obsidian/sync triggers an incremental sync.

    The endpoint is intentionally fire-and-forget — it starts the sync in
    a background thread and returns immediately with ``status=started``.
    Sync progress is observable via the separate ``GET .../sync`` endpoint.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Test note\n\nContent here.")
    app.post("/v1/connectors/obsidian/connect", json={"path": str(vault)})
    resp = app.post("/v1/connectors/obsidian/sync")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connector_id"] == "obsidian"
    assert data["status"] in {"started", "already_syncing"}


# ---------------------------------------------------------------------------
# Connect-time credential validation (GH #409): the /connect endpoint must
# reject invalid credentials with HTTP 400 and never persist (or overwrite)
# anything on disk when validation fails.
# ---------------------------------------------------------------------------


def test_connect_slack_bot_token_returns_400(app, tmp_path: Path) -> None:
    """POST connect with an xoxb- token is rejected 400 and writes nothing."""
    from openjarvis.connectors.slack_connector import SlackConnector
    from openjarvis.server.connectors_router import _instances

    creds = tmp_path / "slack.json"
    _instances["slack"] = SlackConnector(credentials_path=str(creds))
    try:
        resp = app.post(
            "/v1/connectors/slack/connect", json={"token": "xoxb-fake-token"}
        )
        assert resp.status_code == 400
        assert "xoxb" in resp.json()["detail"].lower()
        assert not creds.exists()
    finally:
        _instances.pop("slack", None)


def test_connect_granola_invalid_key_returns_400_keeps_existing(
    app, tmp_path: Path
) -> None:
    """A bad Granola key is rejected 400 and the existing credential survives."""
    import json
    from unittest.mock import patch

    from openjarvis.connectors.granola import GranolaConnector, GranolaKeyError
    from openjarvis.server.connectors_router import _instances

    creds = tmp_path / "granola.json"
    creds.write_text(json.dumps({"token": "grl_real_existing_key"}))
    _instances["granola"] = GranolaConnector(credentials_path=str(creds))
    try:
        with patch(
            "openjarvis.connectors.granola._granola_api_validate_key",
            side_effect=GranolaKeyError(
                "Invalid API key. Check your key in Granola Settings → API."
            ),
        ):
            resp = app.post(
                "/v1/connectors/granola/connect",
                json={"code": "fake-key-12345"},
            )
        assert resp.status_code == 400
        assert "Invalid API key" in resp.json()["detail"]
        # The previously-working credential must be untouched.
        assert json.loads(creds.read_text())["token"] == "grl_real_existing_key"
    finally:
        _instances.pop("granola", None)


@pytest.mark.parametrize(
    ("connector_id", "filename", "payload", "expected"),
    [
        (
            "github_notifications",
            "github.json",
            {"token": "ghp_test"},
            {"token": "ghp_test"},
        ),
        ("oura", "oura.json", {"token": "oura_test"}, {"token": "oura_test"}),
        (
            "weather",
            "weather.json",
            {"token": "weather_key", "config": {"location": "Boston,US"}},
            {"api_key": "weather_key", "location": "Boston,US"},
        ),
    ],
)
def test_connect_persists_generic_token_connector_credentials(
    app,
    tmp_path: Path,
    monkeypatch,
    connector_id: str,
    filename: str,
    payload: dict,
    expected: dict,
) -> None:
    """The generic token panel must actually configure each token connector."""

    from openjarvis.connectors.github_notifications import GitHubNotificationsConnector
    from openjarvis.connectors.oura import OuraConnector
    from openjarvis.connectors.weather import WeatherConnector
    from openjarvis.core.registry import ConnectorRegistry
    from openjarvis.server.connectors_router import _instances

    path = tmp_path / filename
    constructors = {
        "github_notifications": lambda: GitHubNotificationsConnector(
            token_path=str(path)
        ),
        "oura": lambda: OuraConnector(token_path=str(path)),
        "weather": lambda: WeatherConnector(token_path=str(path)),
    }
    instance = constructors[connector_id]()
    ConnectorRegistry.register_value(connector_id, type(instance))
    validators = {
        "github_notifications": (
            "openjarvis.connectors.github_notifications._github_api_get",
            [],
        ),
        "oura": ("openjarvis.connectors.oura._oura_api_get", {}),
        "weather": ("openjarvis.connectors.weather._weather_api_get", {}),
    }
    target, result = validators[connector_id]
    monkeypatch.setattr(target, lambda *args, **kwargs: result)
    # The endpoint deliberately starts an asynchronous initial sync.  Replace
    # it with an empty iterator so this endpoint test never reaches a real API.
    instance.sync = lambda **_kwargs: iter(())
    _instances[connector_id] = instance
    try:
        resp = app.post(f"/v1/connectors/{connector_id}/connect", json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json()["connected"] is True
        assert json.loads(path.read_text()) == expected
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        _instances.pop(connector_id, None)


def test_invalid_token_is_not_persisted_or_synced(app, tmp_path, monkeypatch) -> None:
    """Validation failure preserves an existing credential byte-for-byte."""
    from openjarvis.connectors.github_notifications import (
        GitHubNotificationsConnector,
    )
    from openjarvis.core.registry import ConnectorRegistry
    from openjarvis.server.connectors_router import _instances

    path = tmp_path / "github.json"
    original = '{"token":"known-good"}'
    path.write_text(original, encoding="utf-8")
    instance = GitHubNotificationsConnector(token_path=str(path))
    ConnectorRegistry.register_value("github_notifications", type(instance))
    sync_called = False

    def sync(**kwargs):
        nonlocal sync_called
        sync_called = True
        return iter(())

    instance.sync = sync
    monkeypatch.setattr(
        "openjarvis.connectors.github_notifications._github_api_get",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")),
    )
    _instances["github_notifications"] = instance
    try:
        response = app.post(
            "/v1/connectors/github_notifications/connect",
            json={"token": "bad-token"},
        )
        assert response.status_code == 400
        assert path.read_text(encoding="utf-8") == original
        assert sync_called is False
    finally:
        _instances.pop("github_notifications", None)


def test_connect_weather_requires_location(app, tmp_path: Path) -> None:
    """Weather must not report connected with an API key it cannot use."""
    from openjarvis.connectors.weather import WeatherConnector
    from openjarvis.server.connectors_router import _instances

    path = tmp_path / "weather.json"
    _instances["weather"] = WeatherConnector(token_path=str(path))
    try:
        resp = app.post("/v1/connectors/weather/connect", json={"token": "weather_key"})
        assert resp.status_code == 400
        assert "location" in resp.json()["detail"].lower()
        assert not path.exists()
    finally:
        _instances.pop("weather", None)


def test_connect_news_rss_requires_and_persists_feeds(app, tmp_path: Path) -> None:
    """A local connector with required setup cannot claim success for ``{}``."""
    from openjarvis.connectors.news_rss import NewsRSSConnector
    from openjarvis.server.connectors_router import _instances

    path = tmp_path / "news_rss.json"
    instance = NewsRSSConnector(config_path=str(path))
    instance.sync = lambda **_kwargs: iter(())
    _instances["news_rss"] = instance
    try:
        assert app.post("/v1/connectors/news_rss/connect", json={}).status_code == 400
        resp = app.post(
            "/v1/connectors/news_rss/connect",
            json={"config": {"feeds": [{"url": "https://example.com/feed.xml"}]}},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["connected"] is True
        assert json.loads(path.read_text())["feeds"] == [
            {"name": "example.com", "url": "https://example.com/feed.xml"}
        ]
    finally:
        _instances.pop("news_rss", None)


def _apple_mail_store(tmp_path: Path, account_id: str = "ACCOUNT-ID") -> Path:
    """Build a throwaway Apple Mail store so no live mailbox is ever read."""
    import sqlite3

    version = tmp_path / "Mail" / "V10"
    (version / account_id / "Inbox.mbox" / "Data" / "Messages").mkdir(parents=True)
    db = version / "MailData" / "Envelope Index"
    db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE mailboxes (ROWID INTEGER PRIMARY KEY, url TEXT);"
            "CREATE TABLE messages (ROWID INTEGER PRIMARY KEY, mailbox INTEGER, "
            "deleted INTEGER, date_received INTEGER);"
            f"INSERT INTO mailboxes VALUES (1, 'imap://{account_id}/Inbox');"
        )
    return tmp_path / "Mail"


def test_apple_mail_detail_advertises_the_local_accounts(app, tmp_path: Path) -> None:
    """The UI cannot build a connect request without the account choices."""
    from openjarvis.connectors.apple_mail import AppleMailConnector
    from openjarvis.server.connectors_router import _instances

    root = _apple_mail_store(tmp_path)
    _instances["apple_mail"] = AppleMailConnector(
        str(tmp_path / "apple_mail.json"), str(root)
    )
    try:
        resp = app.get("/v1/connectors/apple_mail")
        assert resp.status_code == 200
        assert resp.json()["setup_options"] == {
            "accounts": [
                {
                    "account_id": "ACCOUNT-ID",
                    "protocol": "imap",
                    "mail_version": "V10",
                }
            ]
        }
    finally:
        _instances.pop("apple_mail", None)


def test_connect_apple_mail_requires_a_known_account(app, tmp_path: Path) -> None:
    """Neither a missing nor an unknown account may silently sync nothing."""
    from openjarvis.connectors.apple_mail import AppleMailConnector
    from openjarvis.server.connectors_router import _instances

    root = _apple_mail_store(tmp_path)
    config = tmp_path / "apple_mail.json"
    _instances["apple_mail"] = AppleMailConnector(str(config), str(root))
    try:
        assert app.post("/v1/connectors/apple_mail/connect", json={}).status_code == 400
        resp = app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "MISSING-ACCOUNT"}},
        )
        assert resp.status_code == 400
        assert "MISSING-ACCOUNT" in resp.json()["detail"]
        assert not config.exists()
    finally:
        _instances.pop("apple_mail", None)


def _apple_mail_accounts(
    tmp_path: Path,
    messages: tuple[tuple[str, str, str, str], ...],
) -> Path:
    """Build a throwaway multi-account Apple Mail store.

    Each entry is ``(account_id, row_id, subject, date_header)``. Nothing here
    touches a live mailbox; the store is a temporary SQLite index plus one
    ``.emlx`` file per message.
    """
    import sqlite3
    from email.message import EmailMessage

    root = tmp_path / "Mail"
    version = root / "V10"
    db = version / "MailData" / "Envelope Index"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE mailboxes (ROWID INTEGER PRIMARY KEY, url TEXT);"
            "CREATE TABLE messages (ROWID INTEGER PRIMARY KEY, mailbox INTEGER, "
            "deleted INTEGER, date_received INTEGER);"
        )
        mailboxes: dict[str, int] = {}
        for account_id, row_id, subject, date_header in messages:
            if account_id not in mailboxes:
                mailboxes[account_id] = len(mailboxes) + 1
                conn.execute(
                    "INSERT INTO mailboxes VALUES (?, ?)",
                    (mailboxes[account_id], f"imap://{account_id}/Inbox"),
                )
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, 0, ?)",
                (int(row_id), mailboxes[account_id], int(row_id)),
            )
            msg = EmailMessage()
            msg["Message-ID"] = f"<{row_id}@example.test>"
            msg["Subject"] = subject
            msg["From"] = "sender@example.test"
            msg["To"] = "recipient@example.test"
            msg["Date"] = date_header
            msg.set_content(subject)
            raw = msg.as_bytes()
            folder = version / account_id / "Inbox.mbox" / "Data" / "Messages"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{row_id}.emlx").write_bytes(
                str(len(raw)).encode() + b"\n" + raw
            )
    return root


# Account A's mail is newer than account B's, which is what makes an inherited
# watermark visible: a ``since`` filter left at A's completion time skips every
# one of B's messages, and the backfill reports a clean zero-document sync.
_ACCOUNT_A_SUBJECT = "alpha account mail"
_ACCOUNT_B_SUBJECT = "bravo account mail"
_ACCOUNT_A_MESSAGE = (
    "ACCOUNT-A",
    "41",
    _ACCOUNT_A_SUBJECT,
    "Wed, 10 Sep 2025 12:00:00 +0000",
)
_ACCOUNT_B_MESSAGE = (
    "ACCOUNT-B",
    "42",
    _ACCOUNT_B_SUBJECT,
    "Sun, 05 Jan 2025 12:00:00 +0000",
)


def _apple_mail_doc_ids() -> set[str]:
    """Return the doc ids the knowledge store holds for the two test accounts.

    The subjects are plain words on purpose: the store searches over an FTS5
    index, where a hyphenated term parses as an operator rather than as text
    and matches nothing.
    """
    from openjarvis.connectors.store import KnowledgeStore

    found = set()
    with KnowledgeStore() as store:
        for subject in (_ACCOUNT_A_SUBJECT, _ACCOUNT_B_SUBJECT):
            for result in store.retrieve(subject, top_k=20, source="apple_mail"):
                doc_id = str(result.metadata.get("doc_id", ""))
                if doc_id:
                    found.add(doc_id)
    return found


def _apple_mail_checkpoint() -> dict | None:
    from openjarvis.connectors.pipeline import IngestionPipeline
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.connectors.sync_engine import SyncEngine

    with KnowledgeStore() as store:
        with SyncEngine(pipeline=IngestionPipeline(store=store)) as engine:
            return engine.get_checkpoint("apple_mail")


def _await_apple_mail_sync(app, timeout: float = 10.0) -> dict:
    """Block until the background sync worker has stopped writing.

    The worker's thread lives in the router's closure, so the polling endpoint
    the UI uses is also the only handle a test has on it.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = app.get("/v1/connectors/apple_mail/sync").json()
        if state["state"] not in {"syncing", "stopping"}:
            return state
        time.sleep(0.01)
    pytest.fail("the apple_mail sync worker never stopped")


def _disconnect_when_the_worker_stops(app, timeout: float = 10.0):
    """Retry the disconnect until the worker it is waiting on has stopped.

    A refused disconnect deliberately leaves the connector marked as stopping:
    its worker still owns the store, and the caller is told to retry. Retrying
    is therefore both the documented way out and the only signal a test has
    that the join finally succeeded.
    """
    deadline = time.time() + timeout
    while True:
        response = app.post("/v1/connectors/apple_mail/disconnect")
        if response.status_code != 409 or time.time() >= deadline:
            return response
        time.sleep(0.01)


@pytest.fixture
def apple_mail_account_a(tmp_path: Path):
    """An Apple Mail connector pinned to account A, with A's mail indexed."""
    from openjarvis.connectors.apple_mail import AppleMailConnector
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.server.connectors_router import _instances

    root = _apple_mail_accounts(tmp_path, (_ACCOUNT_A_MESSAGE, _ACCOUNT_B_MESSAGE))
    config = tmp_path / "apple_mail.json"
    connector = AppleMailConnector(str(config), str(root))
    _instances["apple_mail"] = connector
    try:
        with KnowledgeStore() as store:
            store.delete_by_sources({"apple_mail"})
        _reset_apple_mail_checkpoint()
        yield connector, config
    finally:
        _instances.pop("apple_mail", None)
        with KnowledgeStore() as store:
            store.delete_by_sources({"apple_mail"})
        _reset_apple_mail_checkpoint()


def _reset_apple_mail_checkpoint() -> None:
    from openjarvis.connectors.pipeline import IngestionPipeline
    from openjarvis.connectors.store import KnowledgeStore
    from openjarvis.connectors.sync_engine import SyncEngine

    with KnowledgeStore() as store:
        with SyncEngine(pipeline=IngestionPipeline(store=store)) as engine:
            engine.reset_checkpoint("apple_mail")


def test_disconnect_keeps_the_account_bound_when_the_purge_fails(
    app,
    monkeypatch,
    tmp_path: Path,
    apple_mail_account_a,
) -> None:
    """A failed cleanup must not clear the binding that blocks a repin.

    Revoking credentials first would drop the account id ``configure()``
    compares against, so the caller would see the failure and then be allowed
    to connect a different account on top of the old account's surviving mail.
    """
    from openjarvis.connectors.apple_mail import AppleMailConnector
    from openjarvis.connectors.store import KnowledgeStore

    connector, config = apple_mail_account_a
    assert (
        app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "ACCOUNT-A"}},
        ).status_code
        == 200
    )
    _await_apple_mail_sync(app)
    assert _apple_mail_doc_ids() == {"apple_mail:ACCOUNT-A:<41@example.test>"}
    before = _apple_mail_checkpoint()
    assert before is not None and before["last_sync"]

    def _locked(self, sources):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(KnowledgeStore, "delete_by_sources", _locked)

    resp = app.post("/v1/connectors/apple_mail/disconnect")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "cleanup_failed"
    assert body["connected"] is True
    assert "still connected" in body["detail"]
    # Nothing was deleted, so nothing may be reported as purged.
    assert body["purged_sources"] == []
    assert body["retained_sources"] == ["apple_mail"]

    monkeypatch.undo()

    # The binding survived, so A's surviving mail cannot be shadowed by B.
    assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"
    assert connector.is_connected()
    assert _apple_mail_doc_ids() == {"apple_mail:ACCOUNT-A:<41@example.test>"}

    # The whole checkpoint came back, not only the fields the purge reads. A
    # partially restored one would let the next sync re-read or skip A's mail.
    assert _apple_mail_checkpoint() == before

    repin = app.post(
        "/v1/connectors/apple_mail/connect",
        json={"config": {"account_id": "ACCOUNT-B"}},
    )
    assert repin.status_code == 409
    assert "retry disconnect before reconnecting" in repin.json()["detail"]
    assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"

    # That refusal is in-memory, so it dies with the process. The binding is
    # what survives a restart, and it has to keep refusing on its own. A
    # connector built from scratch over the same config is what the next
    # process sees.
    restarted = AppleMailConnector(str(config), str(tmp_path / "Mail"))
    assert restarted.is_connected()
    with pytest.raises(ValueError, match="already connected to account 'ACCOUNT-A'"):
        restarted.configure("ACCOUNT-B")
    with pytest.raises(ValueError, match="already connected to account 'ACCOUNT-A'"):
        connector.configure("ACCOUNT-B")
    assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"

    # The shortfall is recoverable: once the store is writable the retry
    # completes, and B then backfills its own older mail from nothing.
    assert app.post("/v1/connectors/apple_mail/disconnect").status_code == 200
    assert not config.exists()
    assert _apple_mail_doc_ids() == set()
    assert (
        app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "ACCOUNT-B"}},
        ).status_code
        == 200
    )
    _await_apple_mail_sync(app)
    assert _apple_mail_doc_ids() == {"apple_mail:ACCOUNT-B:<42@example.test>"}


def test_disconnect_reports_a_failed_revoke_without_claiming_the_content_survived(
    app,
    monkeypatch,
    apple_mail_account_a,
) -> None:
    """Past the purge the content is gone, and the report has to say so.

    This is the deliberate fail-closed shortfall: the account stays bound and
    the repin stays refused until a retry releases it. What the endpoint must
    not do is describe the content as preserved, because it is not.
    """
    connector, config = apple_mail_account_a
    assert (
        app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "ACCOUNT-A"}},
        ).status_code
        == 200
    )
    _await_apple_mail_sync(app)
    assert _apple_mail_doc_ids() == {"apple_mail:ACCOUNT-A:<41@example.test>"}

    def _refused(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(connector, "disconnect", _refused)

    resp = app.post("/v1/connectors/apple_mail/disconnect")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "revoke_failed"
    assert "purged" in body["detail"]
    assert "still connected" not in body["detail"]
    # Apple Mail owns its source alone, so the purge really was complete.
    assert body["purged_sources"] == ["apple_mail"]
    assert body["retained_sources"] == []

    monkeypatch.undo()

    # Content gone, binding retained: the repin guard is still the one in force.
    assert _apple_mail_doc_ids() == set()
    assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"
    assert connector.is_connected()
    repin = app.post(
        "/v1/connectors/apple_mail/connect",
        json={"config": {"account_id": "ACCOUNT-B"}},
    )
    assert repin.status_code == 409
    assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"
    with pytest.raises(ValueError, match="already connected to account 'ACCOUNT-A'"):
        connector.configure("ACCOUNT-B")

    # A retry releases the binding, and B starts from an empty store.
    assert app.post("/v1/connectors/apple_mail/disconnect").status_code == 200
    assert not config.exists()
    assert (
        app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "ACCOUNT-B"}},
        ).status_code
        == 200
    )
    _await_apple_mail_sync(app)
    assert _apple_mail_doc_ids() == {"apple_mail:ACCOUNT-B:<42@example.test>"}


def test_disconnect_refuses_while_a_paused_worker_still_owns_the_account(
    app,
    monkeypatch,
    apple_mail_account_a,
) -> None:
    """A sync paused after capturing a message still owns the store.

    Purging around it -- which is what a second process doing its own cleanup
    would do -- lets the worker resume into a store that has already been
    handed to the next account.
    """
    from openjarvis.server import connectors_router

    connector, config = apple_mail_account_a
    connector.configure("ACCOUNT-A")

    captured = threading.Event()
    released = threading.Event()
    real_sync = connector.sync

    def gated_sync(**kwargs):
        for document in real_sync(**kwargs):
            yield document
            captured.set()
            released.wait(timeout=5)

    monkeypatch.setattr(connector, "sync", gated_sync)
    monkeypatch.setattr(connectors_router, "_SYNC_STOP_TIMEOUT_SECONDS", 0.05)

    try:
        assert app.post("/v1/connectors/apple_mail/sync").status_code == 200
        assert captured.wait(timeout=5)

        resp = app.post("/v1/connectors/apple_mail/disconnect")
        assert resp.status_code == 409
        assert "was not purged" in resp.json()["detail"]
        # Nothing was revoked, so the repin guard is still the one in force.
        assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"

        repin = app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "ACCOUNT-B"}},
        )
        assert repin.status_code == 409
        assert json.loads(config.read_text())["account_id"] == "ACCOUNT-A"
    finally:
        released.set()

    # Only once the worker has stopped does the purge run, and the account that
    # follows starts from nothing rather than from A's leftovers.
    assert _disconnect_when_the_worker_stops(app).status_code == 200
    assert not config.exists()
    assert _apple_mail_doc_ids() == set()


def test_a_reconnected_account_never_inherits_the_previous_watermark(
    app,
    apple_mail_account_a,
) -> None:
    """B's backfill must re-read from the start, not from A's last sync.

    B's mail is older than A's, so a surviving ``last_sync`` would filter every
    message of B's out and report a successful, empty first sync.
    """
    connector, config = apple_mail_account_a

    assert (
        app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "ACCOUNT-A"}},
        ).status_code
        == 200
    )
    _await_apple_mail_sync(app)
    assert _apple_mail_checkpoint()["last_sync"] is not None

    assert app.post("/v1/connectors/apple_mail/disconnect").status_code == 200
    assert not config.exists()
    assert _apple_mail_doc_ids() == set()
    checkpoint = _apple_mail_checkpoint()
    assert checkpoint is None or checkpoint["last_sync"] is None

    assert (
        app.post(
            "/v1/connectors/apple_mail/connect",
            json={"config": {"account_id": "ACCOUNT-B"}},
        ).status_code
        == 200
    )
    _await_apple_mail_sync(app)

    assert _apple_mail_doc_ids() == {"apple_mail:ACCOUNT-B:<42@example.test>"}
