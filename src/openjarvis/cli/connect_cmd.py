"""``jarvis connect`` -- manage data source connections."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

# The server cancels the sync worker and waits for it before purging, so this
# has to outlast that join plus the delete.
_DISCONNECT_TIMEOUT_SECONDS = 30.0


def _list_sources(registry: object) -> None:
    """Print a Rich table of registered connectors and their sync status."""
    console = Console()
    items = registry.items()  # type: ignore[attr-defined]

    if not items:
        console.print("[yellow]No connectors registered.[/yellow]")
        return

    table = Table(title="Connected Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")

    for key, connector_cls in items:
        # Try to instantiate with no args to check status (best-effort)
        try:
            instance = connector_cls()
            connected = instance.is_connected()
            status = "connected" if connected else "disconnected"
            auth_type = getattr(connector_cls, "auth_type", "unknown")
        except Exception:  # noqa: BLE001
            status = "unknown"
            auth_type = getattr(connector_cls, "auth_type", "unknown")

        table.add_row(key, auth_type, status)

    console.print(table)


def _server_base_url() -> str:
    """Return the base URL of the local API server from the saved config."""
    from openjarvis.core.config import load_config

    server = load_config().server
    host = server.host or "127.0.0.1"
    # A wildcard is a bind address, not somewhere to send a request.
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    return f"http://{host}:{server.port}"


def _response_payload(response: object) -> dict:
    """Return a response body as a dict, or an empty one if it is not JSON."""
    try:
        payload = response.json()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _error_detail(response: object) -> str:
    """Pull the FastAPI ``detail`` out of an error response."""
    detail = _response_payload(response).get("detail")
    if detail:
        return str(detail)
    return str(getattr(response, "text", "")).strip()


def _retained_sources(payload: dict) -> list[str] | None:
    """Read the sources the server says it deliberately left indexed.

    Returns the retained source names, an empty list when the server states
    it kept nothing, and ``None`` when the body does not say. A source another
    integration may also own is retained on purpose, so this is the difference
    between "the content is gone" and "most of it is". Absent or malformed
    means unknown, never empty: reading a missing field as "nothing retained"
    is how a partial purge gets reported as a complete one.

    The field says what stayed, not why. The server retains a shared source
    both when it confirms another owner is connected and when that owner's
    connectivity cannot be read at all, and it does not distinguish the two
    here, so nothing downstream may assert a connected peer.
    """
    if "retained_sources" not in payload:
        return None
    retained = payload.get("retained_sources")
    if not isinstance(retained, list) or not all(
        isinstance(name, str) for name in retained
    ):
        return None
    return retained


def _unknown_outcome(console: Console, source: str, reason: str) -> int:
    """Report that the disconnect's effect on the server cannot be established.

    Nothing here may claim that the account survived or that its content was
    kept. The request may have been carried out in full before the answer went
    missing, so the only honest report is that the outcome is unknown and has
    to be read back from the server.
    """
    console.print(f"[red]Could not confirm the {source} disconnect: {reason}[/red]")
    console.print(
        f"[yellow]The request may already have been carried out, so the state of "
        f"{source} is unknown. Check it with 'jarvis connect --list' and run "
        "this again if it is still connected; a repeated disconnect is "
        "safe.[/yellow]"
    )
    return 1


def _disconnect_source(registry: object, source: str) -> int:
    """Disconnect a source through the server that owns its sync worker.

    Returns a process exit status: ``0`` only for a disconnect the server
    confirmed it completed, and ``1`` for refused, failed and unknown
    outcomes, so a script can tell them apart without reading the text.

    This command has no sync thread, but the running service does, and both
    reach the same database. Clearing credentials and deleting rows from here
    races that worker: a sync paused mid-document resumes after the purge,
    re-ingests the account that was just disconnected and rewrites its
    checkpoint, so the next account starts from a watermark that skips its own
    mail. Only the server can cancel the worker, wait for it, and purge under
    its own lifecycle lock, so this asks it to and reports what it answered.

    There is deliberately no local fallback, on any outcome. Without the server
    there is no way to establish that nothing else owns the store, and a
    disconnect that is refused can be retried, where one that half-applied
    cannot be undone.
    """
    import httpx

    console = Console()

    if not registry.contains(source):  # type: ignore[attr-defined]
        console.print(f"[red]Unknown source: {source}[/red]")
        return 1

    base_url = _server_base_url()
    try:
        response = httpx.post(
            f"{base_url}/v1/connectors/{source}/disconnect",
            timeout=_DISCONNECT_TIMEOUT_SECONDS,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        # The connection was never established, so the server never saw the
        # request. This is the one transport failure where "this attempt
        # changed nothing" is a fact rather than a hope. It is not evidence
        # about the binding itself, which nothing here reads.
        console.print(
            f"[red]Could not reach the OpenJarvis server at {base_url}: {exc}[/red]"
        )
        console.print(
            "[yellow]The request was never sent, so this attempt changed "
            "nothing: no credentials were cleared and nothing was deleted. It "
            f"says nothing about whether {source} is bound to an account right "
            "now, which is not checked here. Disconnecting runs in the server "
            "that owns the sync worker — start it with 'jarvis serve' and try "
            "again.[/yellow]"
        )
        return 1
    except httpx.HTTPError as exc:
        # A read timeout or a dropped connection happens after the request has
        # gone out. The purge may have run to completion.
        return _unknown_outcome(console, source, f"{type(exc).__name__}: {exc}")

    payload = _response_payload(response)
    if response.status_code == 200:
        # Both fields have to be there and correctly typed. A body that omits
        # ``connected``, or sends ``null``, ``0`` or ``""`` for it, has
        # confirmed nothing; reading those as falsy would turn a malformed
        # answer into a success. Only the literal ``false`` is a confirmation.
        if (
            payload.get("status") == "disconnected"
            and payload.get("connected") is False
        ):
            retained = _retained_sources(payload)
            if retained is None:
                # The lifecycle finished, so this is a success, but the body
                # did not say what the purge covered. Claiming the content is
                # gone would be inventing the part the server left out.
                console.print(f"[green]Disconnected {source}.[/green]")
                console.print(
                    "[yellow]The server did not report which indexed content "
                    "was removed, so this cannot say whether any of it was "
                    "kept. Check with 'jarvis connect --list'.[/yellow]"
                )
                return 0
            if retained:
                # A source another integration also writes to is retained on
                # purpose. Those rows are still indexed, and the message has to
                # say so rather than claim a full purge. It says ownership may
                # be shared, because the server retains on an unreadable peer
                # probe too and this field cannot tell the two apart.
                console.print(f"[green]Disconnected {source}.[/green]")
                console.print(
                    "[yellow]Indexed content under "
                    + ", ".join(retained)
                    + " was kept because ownership may be shared with another "
                    "integration. Everything else this source indexed was "
                    "removed.[/yellow]"
                )
                return 0
            console.print(
                f"[green]Disconnected {source} and purged its indexed content.[/green]"
            )
            return 0
        # A 200 whose body does not agree that the lifecycle finished is not a
        # success, and it is not evidence of preservation either.
        return _unknown_outcome(
            console, source, "the server returned 200 with an unrecognised body"
        )

    detail = _error_detail(response) or f"HTTP {response.status_code}"
    if response.status_code == 409:
        console.print(f"[yellow]{source} is still stopping: {detail}[/yellow]")
        console.print(
            "[yellow]It stays connected and nothing was purged. Run this again "
            "once the sync has stopped.[/yellow]"
        )
        return 1

    code = str(payload.get("code", ""))
    if code == "cleanup_failed":
        console.print(f"[red]Could not disconnect {source}: {detail}[/red]")
        console.print(
            f"[yellow]{source} is still connected. Its indexed content was left "
            "in place, so reconnecting a different account stays blocked until "
            "this succeeds.[/yellow]"
        )
        return 1

    if code == "revoke_failed":
        console.print(f"[red]Could not disconnect {source}: {detail}[/red]")
        retained = _retained_sources(payload)
        if retained is None:
            # The purge ran, but the body did not say what it covered, so the
            # scope of what is gone cannot be stated here.
            console.print(
                f"[yellow]The purge ran, but the server did not report what it "
                f"covered, so how much indexed content is left is unknown. "
                f"{source} is still bound to its account, so reconnecting a "
                "different one stays blocked until a retry succeeds.[/yellow]"
            )
        elif retained:
            console.print(
                "[yellow]Indexed content under "
                + ", ".join(retained)
                + " was kept because ownership may be shared with another "
                f"integration; the rest is gone. {source} is still bound to its "
                "account, so reconnecting a different one stays blocked until "
                "a retry succeeds.[/yellow]"
            )
        else:
            console.print(
                f"[yellow]The indexed content is gone, but {source} is still "
                "bound to its account, so reconnecting a different one stays "
                "blocked until a retry succeeds.[/yellow]"
            )
        return 1

    # An error the server did not label. It may have failed before the purge or
    # after it, so neither the content nor the binding can be spoken for.
    return _unknown_outcome(console, source, detail)


def _connect_apple_mail(connector_cls: type, account: str) -> None:
    """Pin the Apple Mail connector to one locally cached account."""
    console = Console()
    instance = connector_cls()

    try:
        accounts = instance.available_accounts()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not read the local Apple Mail store: {exc}[/red]")
        return

    if not accounts:
        console.print(
            "[red]No local Apple Mail accounts found. Open Mail.app at least once, "
            "and grant Full Disk Access to your terminal.[/red]"
        )
        return

    if not account:
        table = Table(title="Local Apple Mail accounts")
        table.add_column("Account ID", style="cyan")
        table.add_column("Protocol", style="magenta")
        table.add_column("Store", style="green")
        for entry in accounts:
            table.add_row(entry["account_id"], entry["protocol"], entry["mail_version"])
        console.print(table)
        console.print(
            "[yellow]Re-run with --account <id> to index one account.[/yellow]"
        )
        return

    try:
        instance.configure(account)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    except OSError as exc:
        console.print(f"[red]Could not save the Apple Mail configuration: {exc}[/red]")
        return

    if instance.is_connected():
        console.print(f"[green]apple_mail connected to account {account}.[/green]")
    else:
        console.print(
            f"[red]apple_mail saved account {account} but its local store is "
            "unreadable.[/red]"
        )


def _connect_source(
    registry: object, source: str, path: str = "", account: str = ""
) -> None:
    """Route connector setup by auth_type."""
    console = Console()

    if not registry.contains(source):  # type: ignore[attr-defined]
        console.print(f"[red]Unknown source: {source}[/red]")
        console.print(
            "[yellow]Available sources: "
            + ", ".join(registry.keys())  # type: ignore[attr-defined]
            + "[/yellow]"
        )
        return

    connector_cls = registry.get(source)  # type: ignore[attr-defined]
    auth_type = getattr(connector_cls, "auth_type", "")

    if source == "apple_mail":
        _connect_apple_mail(connector_cls, account)

    elif auth_type == "filesystem":
        # Filesystem connectors (e.g. Obsidian) need a path
        if not path:
            console.print(
                f"[red]{source} requires a --path argument (e.g. --path ~/vault).[/red]"
            )
            return
        try:
            instance = connector_cls(vault_path=path)
        except TypeError:
            try:
                instance = connector_cls(path)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Failed to create {source} connector: {exc}[/red]")
                return

        if instance.is_connected():
            console.print(f"[green]{source} connected at path: {path}[/green]")
        else:
            console.print(
                f"[red]{source}: path '{path}' does not exist or is not accessible."
                "[/red]"
            )

    elif auth_type == "oauth":
        # OAuth connectors — auto-open browser + catch callback
        from openjarvis.connectors.oauth import (
            get_client_credentials,
            get_provider_for_connector,
            run_connector_oauth,
            save_client_credentials,
        )

        try:
            instance = connector_cls()
            if instance.is_connected():
                console.print(f"[green]{source} is already connected.[/green]")
                return

            provider = get_provider_for_connector(source)
            if provider is None:
                console.print(f"[red]No OAuth provider configured for {source}.[/red]")
                return

            creds = get_client_credentials(provider)
            client_id = creds[0] if creds else ""
            client_secret = creds[1] if creds else ""

            if not client_id or not client_secret:
                console.print(f"[cyan]First-time setup for {source}.[/cyan]")
                console.print(
                    f"[yellow]Create an OAuth app at: {provider.setup_url}[/yellow]"
                )
                console.print(f"[dim]{provider.setup_hint}[/dim]")
                client_id = click.prompt("Client ID")
                client_secret = click.prompt("Client Secret")
                save_client_credentials(provider, client_id, client_secret)

            run_connector_oauth(source, client_id, client_secret)
            console.print(f"[green]{source} authorised successfully.[/green]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]OAuth flow failed for {source}: {exc}[/red]")

    elif auth_type == "token":
        # Token-based connectors (e.g. Oura) — prompt for personal access token
        import json
        from pathlib import Path

        from openjarvis.connectors.oauth import save_tokens
        from openjarvis.core.config import DEFAULT_CONFIG_DIR

        try:
            instance = connector_cls()
            if instance.is_connected():
                console.print(f"[green]{source} is already connected.[/green]")
                return

            token = click.prompt(f"Enter your {source} personal access token")
            token_dir = Path(DEFAULT_CONFIG_DIR) / "connectors"
            token_dir.mkdir(parents=True, exist_ok=True)
            token_file = token_dir / f"{source}.json"
            token_file.write_text(json.dumps({"token": token}))
            save_tokens(source, {"token": token})
            console.print(f"[green]{source} connected successfully.[/green]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Token setup failed for {source}: {exc}[/red]")

    else:
        # Generic / bridge connectors
        try:
            instance = connector_cls()
            connected = instance.is_connected()
            status = "connected" if connected else "disconnected"
            console.print(f"{source} status: {status}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to connect {source}: {exc}[/red]")


@click.group(invoke_without_command=True)
@click.argument("source", required=False)
@click.option(
    "--list",
    "list_sources",
    is_flag=True,
    help="List connected sources and sync status.",
)
@click.option(
    "--sync",
    "trigger_sync",
    is_flag=True,
    help="Trigger incremental sync for all sources.",
)
@click.option(
    "--disconnect",
    "disconnect_source",
    default="",
    help="Disconnect a source.",
)
@click.option(
    "--path",
    default="",
    help="Path for filesystem connectors (e.g., Obsidian vault).",
)
@click.option(
    "--account",
    default="",
    help="Account id for account-scoped local connectors (e.g., apple_mail). "
    "Omit to list the accounts available locally.",
)
@click.pass_context
def connect(
    ctx: click.Context,
    source: str | None,
    list_sources: bool,
    trigger_sync: bool,
    disconnect_source: str,
    path: str,
    account: str,
) -> None:
    """Manage data source connections (Gmail, Obsidian, etc.)."""
    # Lazy imports to avoid top-level side effects
    import openjarvis.connectors  # noqa: F401 — registers all connectors
    from openjarvis.core.registry import ConnectorRegistry

    if list_sources:
        _list_sources(ConnectorRegistry)
        return

    if trigger_sync:
        click.echo("Sync not yet implemented in CLI")
        return

    if disconnect_source:
        # Pending, failed and unknown disconnects exit non-zero so a caller can
        # branch on the status rather than on the printed text.
        ctx.exit(_disconnect_source(ConnectorRegistry, disconnect_source))

    if source:
        _connect_source(ConnectorRegistry, source, path=path, account=account)
        return

    # No arguments — show help
    click.echo(ctx.get_help())
