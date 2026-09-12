from __future__ import annotations

import asyncio
import json
import subprocess
from unittest.mock import MagicMock

import pytest

from openjarvis.core.types import Message, Role
from openjarvis.engine._base import EngineConnectionError
from openjarvis.engine.codex_cli import CodexCLIEngine, _parse_events


def _installed(monkeypatch, tmp_path, *, filesystem: bool = True) -> None:
    """Pretend codex is installed, with or without the filesystem opt-in."""
    monkeypatch.setattr(
        "openjarvis.engine.codex_cli.shutil.which", lambda _: "/bin/codex"
    )
    monkeypatch.setenv("OPENJARVIS_CODEX_CWD", str(tmp_path))
    if filesystem:
        monkeypatch.setenv("OPENJARVIS_FILESYSTEM_MODE", "read-write")
    else:
        monkeypatch.delenv("OPENJARVIS_FILESYSTEM_MODE", raising=False)


def _completed(text: str = "answer") -> str:
    return json.dumps(
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
    )


def test_parse_events_extracts_message_and_usage() -> None:
    output = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t"}),
            _completed("hello"),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                }
            ),
        ]
    )

    assert _parse_events(output) == (
        "hello",
        {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    )


def test_parse_events_accepts_tool_items_the_opted_in_turn_may_produce() -> None:
    output = "\n".join(
        [
            json.dumps(
                {"type": "item.completed", "item": {"type": "command_execution"}}
            ),
            _completed(),
        ]
    )

    assert _parse_events(output)[0] == "answer"


def test_text_only_generate_refuses_before_launching_a_child(
    monkeypatch, tmp_path
) -> None:
    """The refusal is the whole enforcement: nothing may run and then be judged."""
    _installed(monkeypatch, tmp_path, filesystem=False)
    popen = MagicMock()
    monkeypatch.setattr("openjarvis.engine.codex_cli.subprocess.Popen", popen)

    with pytest.raises(EngineConnectionError) as excinfo:
        CodexCLIEngine().generate(
            [Message(role=Role.USER, content="hello")], model="codex/default"
        )

    popen.assert_not_called()
    assert "OPENJARVIS_FILESYSTEM_MODE=read-write" in str(excinfo.value)
    assert "claude_cli" in str(excinfo.value)


@pytest.mark.asyncio
async def test_text_only_stream_refuses_before_launching_a_child(
    monkeypatch, tmp_path
) -> None:
    _installed(monkeypatch, tmp_path, filesystem=False)
    exec_mock = MagicMock()
    monkeypatch.setattr(
        "openjarvis.engine.codex_cli.asyncio.create_subprocess_exec", exec_mock
    )

    with pytest.raises(EngineConnectionError, match="OPENJARVIS_FILESYSTEM_MODE"):
        async for _ in CodexCLIEngine().stream([], model="codex/default"):
            pass

    exec_mock.assert_not_called()


def test_text_only_health_is_false_without_probing(monkeypatch, tmp_path) -> None:
    """Discovery must not advertise an engine that cannot serve a turn."""
    _installed(monkeypatch, tmp_path, filesystem=False)
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("openjarvis.engine.codex_cli.subprocess.run", run)
    engine = CodexCLIEngine()

    assert engine.health() is False
    assert engine.list_models() == []
    run.assert_not_called()


def test_generate_uses_hardened_codex_command(monkeypatch, tmp_path) -> None:
    process = MagicMock()
    process.communicate.return_value = (_completed(), None)
    process.returncode = 0
    popen = MagicMock(return_value=process)
    _installed(monkeypatch, tmp_path)
    monkeypatch.setattr("openjarvis.engine.codex_cli.subprocess.Popen", popen)
    engine = CodexCLIEngine()

    result = engine.generate(
        [Message(role=Role.USER, content="hello")], model="codex/default"
    )

    command = popen.call_args.args[0]
    assert result["content"] == "answer"
    assert "--ephemeral" in command
    assert ["--sandbox", "workspace-write"] == command[
        command.index("--sandbox") : command.index("--sandbox") + 2
    ]
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--model" not in command
    assert popen.call_args.kwargs["stderr"] is subprocess.DEVNULL
    prompt = process.communicate.call_args.args[0]
    assert f"only inside {tmp_path}" in prompt
    assert '"content": "hello"' in prompt


def test_generate_hides_provider_credentials_from_the_child(
    monkeypatch, tmp_path
) -> None:
    process = MagicMock()
    process.communicate.return_value = (_completed(), None)
    process.returncode = 0
    popen = MagicMock(return_value=process)
    _installed(monkeypatch, tmp_path)
    monkeypatch.setattr("openjarvis.engine.codex_cli.subprocess.Popen", popen)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CODEX_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://redirect.example.test")
    # The subscription credential lives under CODEX_HOME, so stripping it would
    # break the very isolation this scrub exists to protect.
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    CodexCLIEngine().generate(
        [Message(role=Role.USER, content="hello")], model="codex/default"
    )

    env = popen.call_args.kwargs["env"]
    assert "OPENAI_API_KEY" not in env
    assert "CODEX_API_KEY" not in env
    assert "OPENAI_BASE_URL" not in env
    assert env["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert popen.call_args.kwargs["cwd"] == tmp_path


def test_health_check_also_hides_provider_credentials(monkeypatch, tmp_path) -> None:
    _installed(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("openjarvis.engine.codex_cli.subprocess.run", run)

    assert CodexCLIEngine().health() is True
    assert "OPENAI_API_KEY" not in run.call_args.kwargs["env"]


def test_filesystem_mode_uses_only_configured_workspace(monkeypatch, tmp_path) -> None:
    _installed(monkeypatch, tmp_path)
    engine = CodexCLIEngine()

    command = engine._command("codex/default")
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("-C") + 1] == str(tmp_path)
    assert f"only inside {tmp_path}" in engine._prompt([])


def test_github_mode_advertises_authenticated_cli(monkeypatch, tmp_path) -> None:
    _installed(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_GITHUB_MODE", "read-write")

    assert "authenticated GitHub CLI" in CodexCLIEngine()._prompt([])


def test_generate_kills_process_group_on_timeout(monkeypatch, tmp_path) -> None:
    process = MagicMock()
    process.communicate.side_effect = subprocess.TimeoutExpired("codex", 1)
    process.pid = 42
    _installed(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "openjarvis.engine.codex_cli.subprocess.Popen", lambda *a, **k: process
    )
    killpg = MagicMock()
    monkeypatch.setattr("openjarvis.engine.codex_cli.os.killpg", killpg)

    with pytest.raises(EngineConnectionError, match="timed out"):
        CodexCLIEngine(timeout=1).generate([], model="codex/default")

    killpg.assert_called_once_with(42, 9)
    process.wait.assert_called_once()


def test_health_requires_successful_chatgpt_login(monkeypatch, tmp_path) -> None:
    _installed(monkeypatch, tmp_path)
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("openjarvis.engine.codex_cli.subprocess.run", run)

    assert CodexCLIEngine().health() is True
    assert run.call_args.args[0] == ["/bin/codex", "login", "status"]
    assert run.call_args.kwargs["stdout"] is subprocess.DEVNULL


def test_can_serve_only_codex_models(monkeypatch, tmp_path) -> None:
    _installed(monkeypatch, tmp_path)
    engine = CodexCLIEngine()

    assert engine.can_serve("codex/default")
    assert engine.can_serve("codex/gpt-5")
    assert not engine.can_serve("gpt-5")


def _stream_process(
    lines: list[bytes], *, returncode: int | None = 0, exit_code: int = 0
) -> MagicMock:
    """A create_subprocess_exec stand-in that replays ``lines`` then EOF.

    ``returncode=None`` models a child that is still running, which is what
    the cleanup path has to deal with.
    """
    process = MagicMock()
    process.pid = 42
    process.returncode = returncode
    process.stdin = MagicMock()
    process.stdin.drain = MagicMock(side_effect=lambda: asyncio.sleep(0))
    process.stdout = MagicMock()
    replies = [*lines, b""]
    process.stdout.readline = MagicMock(
        side_effect=lambda: asyncio.sleep(0, result=replies.pop(0))
    )
    process.wait = MagicMock(side_effect=lambda: asyncio.sleep(0, result=exit_code))
    return process


def _streaming(monkeypatch, tmp_path, process: MagicMock) -> MagicMock:
    """Install *process* as the streamed child and return the killpg spy."""
    _installed(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "openjarvis.engine.codex_cli.asyncio.create_subprocess_exec",
        MagicMock(return_value=asyncio.sleep(0, result=process)),
    )
    killpg = MagicMock()
    monkeypatch.setattr("openjarvis.engine.codex_cli.os.killpg", killpg)
    return killpg


@pytest.mark.asyncio
async def test_stream_hides_provider_credentials_from_the_child(
    monkeypatch, tmp_path
) -> None:
    process = _stream_process([_completed().encode()])
    exec_mock = MagicMock(return_value=asyncio.sleep(0, result=process))
    _installed(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "openjarvis.engine.codex_cli.asyncio.create_subprocess_exec", exec_mock
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")

    chunks = [
        chunk async for chunk in CodexCLIEngine().stream([], model="codex/default")
    ]

    assert chunks == ["answer"]
    assert "OPENAI_API_KEY" not in exec_mock.call_args.kwargs["env"]
    assert exec_mock.call_args.kwargs["cwd"] == tmp_path


@pytest.mark.asyncio
async def test_stream_reaps_the_child_when_the_consumer_stops_early(
    monkeypatch, tmp_path
) -> None:
    """Closing the generator at a yield must not leave the child running.

    A caller that disconnects mid-turn abandons the generator; without the
    cleanup the codex child keeps the workspace for the rest of its own life.
    """
    process = _stream_process([_completed().encode()], returncode=None)
    killpg = _streaming(monkeypatch, tmp_path, process)

    stream = CodexCLIEngine().stream([], model="codex/default")
    assert await stream.__anext__() == "answer"
    await stream.aclose()

    killpg.assert_called_once_with(42, 9)
    process.wait.assert_called_once()


@pytest.mark.asyncio
async def test_stream_reaps_the_child_when_the_turn_fails(
    monkeypatch, tmp_path
) -> None:
    """A ``turn.failed`` event leaves the loop without reading EOF."""
    failed = json.dumps({"type": "turn.failed"}).encode()
    process = _stream_process([failed], returncode=None)
    killpg = _streaming(monkeypatch, tmp_path, process)

    with pytest.raises(EngineConnectionError, match="turn failed"):
        async for _ in CodexCLIEngine().stream([], model="codex/default"):
            pass

    killpg.assert_called_once_with(42, 9)


@pytest.mark.asyncio
async def test_stream_bounds_the_prompt_write(monkeypatch, tmp_path) -> None:
    """A child that never reads stdin must not hang the turn in ``drain()``."""
    process = _stream_process([_completed().encode()], returncode=None)
    process.stdin.drain = MagicMock(side_effect=lambda: asyncio.sleep(30))
    killpg = _streaming(monkeypatch, tmp_path, process)

    with pytest.raises(EngineConnectionError, match="timed out"):
        async for _ in CodexCLIEngine(timeout=0.01).stream([], model="codex/default"):
            pass

    killpg.assert_called_once_with(42, 9)


@pytest.mark.asyncio
async def test_stream_deadline_is_absolute_across_the_whole_turn(
    monkeypatch, tmp_path
) -> None:
    """The budget covers the turn, not each read.

    A child that keeps printing can never exceed a per-read timeout, so only
    one deadline for the whole stream actually bounds it.
    """
    clock = {"now": 0.0}
    monkeypatch.setattr("openjarvis.engine.codex_cli.monotonic", lambda: clock["now"])
    lines = [_completed(f"chunk-{index}").encode() for index in range(4)]
    process = _stream_process(lines, returncode=None)
    readline = process.stdout.readline.side_effect

    def _slow_readline():
        # ``wait_for(coro, remaining())`` evaluates the coroutine first, so the
        # time this read costs is what the following remaining() call sees.
        clock["now"] += 4.0
        return readline()

    process.stdout.readline = MagicMock(side_effect=_slow_readline)
    killpg = _streaming(monkeypatch, tmp_path, process)

    chunks = []
    with pytest.raises(EngineConnectionError, match="timed out"):
        async for chunk in CodexCLIEngine(timeout=10).stream([], model="codex/default"):
            chunks.append(chunk)

    assert chunks == ["chunk-0", "chunk-1"]
    killpg.assert_called_once_with(42, 9)


@pytest.mark.asyncio
async def test_stream_deadline_covers_the_final_wait(monkeypatch, tmp_path) -> None:
    """A child that streams its answer and then never exits still times out."""
    clock = {"now": 0.0}
    monkeypatch.setattr("openjarvis.engine.codex_cli.monotonic", lambda: clock["now"])
    process = _stream_process([_completed().encode()], returncode=None)

    def _never_exits():
        clock["now"] += 1000.0
        return asyncio.sleep(0, result=0)

    process.wait = MagicMock(side_effect=_never_exits)
    killpg = _streaming(monkeypatch, tmp_path, process)

    with pytest.raises(EngineConnectionError, match="timed out"):
        async for _ in CodexCLIEngine(timeout=10).stream([], model="codex/default"):
            pass

    killpg.assert_called_once_with(42, 9)


@pytest.mark.asyncio
async def test_stream_cancellation_kills_process(monkeypatch, tmp_path) -> None:
    process = MagicMock()
    process.pid = 42
    process.returncode = None
    process.stdin = MagicMock()
    process.stdin.drain = MagicMock(side_effect=lambda: asyncio.sleep(0))
    process.stdout = MagicMock()
    process.stdout.readline = MagicMock(side_effect=asyncio.CancelledError)
    process.wait = MagicMock(side_effect=lambda: asyncio.sleep(0))
    _installed(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "openjarvis.engine.codex_cli.asyncio.create_subprocess_exec",
        MagicMock(return_value=asyncio.sleep(0, result=process)),
    )
    killpg = MagicMock()
    monkeypatch.setattr("openjarvis.engine.codex_cli.os.killpg", killpg)

    with pytest.raises(asyncio.CancelledError):
        async for _ in CodexCLIEngine().stream([], model="codex/default"):
            pass

    killpg.assert_called_once_with(42, 9)
