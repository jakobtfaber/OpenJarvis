from __future__ import annotations

import asyncio
import json
import subprocess
from unittest.mock import MagicMock

import pytest

from openjarvis.core.types import Message, Role
from openjarvis.engine._base import EngineConnectionError
from openjarvis.engine.claude_cli import ClaudeCLIEngine, _parse_result


def _success(content: str = "answer") -> str:
    return json.dumps(
        {
            "result": content,
            "subtype": "success",
            "is_error": False,
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 3,
                "output_tokens": 4,
            },
        }
    )


def test_parse_result_extracts_message_and_usage() -> None:
    assert _parse_result(_success("hello")) == (
        "hello",
        {
            "prompt_tokens": 15,
            "completion_tokens": 4,
            "total_tokens": 19,
            "cached_input_tokens": 3,
        },
    )


def test_parse_result_rejects_errors_and_invalid_output() -> None:
    with pytest.raises(EngineConnectionError, match="invalid JSON"):
        _parse_result("not json")
    with pytest.raises(EngineConnectionError, match="turn failed"):
        _parse_result(json.dumps({"is_error": True, "subtype": "error"}))


def test_generate_uses_isolated_subscription_command(monkeypatch, tmp_path) -> None:
    process = MagicMock()
    process.communicate.return_value = (_success(), None)
    process.returncode = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setattr("openjarvis.engine.claude_cli.subprocess.Popen", popen)
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setattr(ClaudeCLIEngine, "_subscription_authenticated", lambda _: True)

    result = ClaudeCLIEngine().generate(
        [Message(role=Role.USER, content="hello")], model="claude/default"
    )

    command = popen.call_args.args[0]
    assert result["content"] == "answer"
    for flag in (
        "--safe-mode",
        "--restricted",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--no-chrome",
        "--system-prompt",
    ):
        assert flag in command
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--max-turns") + 1] == "1"
    assert "--add-dir" not in command
    assert "--model" not in command
    assert popen.call_args.kwargs["stderr"] is subprocess.DEVNULL
    assert "ANTHROPIC_API_KEY" not in popen.call_args.kwargs["env"]
    assert "CLAUDE_CODE_USE_VERTEX" not in popen.call_args.kwargs["env"]
    assert '"content": "hello"' in process.communicate.call_args.args[0]
    assert popen.call_args.kwargs["cwd"] == tmp_path


def test_filesystem_mode_enables_scoped_file_tools(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setenv("OPENJARVIS_FILESYSTEM_MODE", "read-write")

    command = ClaudeCLIEngine()._command("claude/default")
    assert command[command.index("--tools") + 1] == "Read,Write,Edit,Glob,Grep"
    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    assert command[command.index("--max-turns") + 1] == "8"
    # --restricted confines the file tools to the working directories, so the
    # boundary is the CLI's, not just the system prompt's.
    assert "--restricted" in command
    assert command[command.index("--add-dir") + 1] == str(tmp_path)
    assert f"only inside {tmp_path}" in command[command.index("--system-prompt") + 1]


def test_github_mode_alone_does_not_unlock_shell_tools(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.delenv("OPENJARVIS_FILESYSTEM_MODE", raising=False)
    monkeypatch.setenv("OPENJARVIS_GITHUB_MODE", "read-write")

    command = ClaudeCLIEngine()._command("claude/default")
    # Regression: the tool list used to be built by appending ",Bash" to an
    # empty string, which both produced a leading comma and handed a text-only
    # turn a command-running tool.
    assert command[command.index("--tools") + 1] == ""
    assert "--allowed-tools" not in command


def test_github_mode_allows_only_git_and_gh_shell_commands(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setenv("OPENJARVIS_FILESYSTEM_MODE", "read-write")
    monkeypatch.setenv("OPENJARVIS_GITHUB_MODE", "read-write")

    command = ClaudeCLIEngine()._command("claude/default")
    assert command[command.index("--tools") + 1] == ("Read,Write,Edit,Glob,Grep,Bash")
    allowed_index = command.index("--allowed-tools")
    assert command[allowed_index + 1 : allowed_index + 3] == [
        "Bash(gh *)",
        "Bash(git *)",
    ]


def test_named_model_is_forwarded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    command = ClaudeCLIEngine()._command("claude/sonnet")
    assert command[command.index("--model") + 1] == "sonnet"


def test_generate_kills_process_group_on_timeout(monkeypatch, tmp_path) -> None:
    process = MagicMock()
    process.communicate.side_effect = subprocess.TimeoutExpired("claude", 1)
    process.pid = 42
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.subprocess.Popen", lambda *a, **k: process
    )
    killpg = MagicMock()
    monkeypatch.setattr("openjarvis.engine.claude_cli.os.killpg", killpg)
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setattr(ClaudeCLIEngine, "_subscription_authenticated", lambda _: True)

    with pytest.raises(EngineConnectionError, match="timed out"):
        ClaudeCLIEngine(timeout=1).generate([], model="claude/default")

    killpg.assert_called_once_with(42, 9)
    process.wait.assert_called_once()


def test_health_requires_claude_subscription(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-auth-check")
    run = MagicMock(
        return_value=MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "subscriptionType": "max",
                }
            ),
        )
    )
    monkeypatch.setattr("openjarvis.engine.claude_cli.subprocess.run", run)

    assert ClaudeCLIEngine().health() is True
    assert run.call_args.args[0] == ["/bin/claude", "auth", "status", "--json"]
    assert "ANTHROPIC_API_KEY" not in run.call_args.kwargs["env"]


def test_oauth_token_is_loaded_only_from_owner_only_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "ambient-token-must-not-pass")
    token_file = tmp_path / "oauth-token"
    token_file.write_text("subscription-token\n")
    token_file.chmod(0o600)
    monkeypatch.setenv("OPENJARVIS_CLAUDE_OAUTH_TOKEN_FILE", str(token_file))

    assert ClaudeCLIEngine()._environment()["CLAUDE_CODE_OAUTH_TOKEN"] == (
        "subscription-token"
    )

    token_file.chmod(0o644)
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in ClaudeCLIEngine()._environment()


def test_health_accepts_protected_subscription_oauth_token(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    token_file = tmp_path / "oauth-token"
    token_file.write_text("subscription-token\n")
    token_file.chmod(0o600)
    monkeypatch.setenv("OPENJARVIS_CLAUDE_OAUTH_TOKEN_FILE", str(token_file))
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.subprocess.run",
        MagicMock(
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps({"loggedIn": True, "authMethod": "oauth_token"}),
            )
        ),
    )

    assert ClaudeCLIEngine().health() is True
    token_file.chmod(0o644)
    assert ClaudeCLIEngine().health() is False


def test_list_models_does_not_repeat_auth_probe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    engine = ClaudeCLIEngine()
    monkeypatch.setattr(
        engine,
        "_subscription_authenticated",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected auth probe")),
    )

    assert engine.list_models() == ["claude/default"]


def test_generate_refuses_non_subscription_auth(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setattr(ClaudeCLIEngine, "_subscription_authenticated", lambda _: False)
    popen = MagicMock()
    monkeypatch.setattr("openjarvis.engine.claude_cli.subprocess.Popen", popen)

    with pytest.raises(EngineConnectionError, match="subscription login"):
        ClaudeCLIEngine().generate([], model="claude/default")

    popen.assert_not_called()


@pytest.mark.asyncio
async def test_stream_cancellation_kills_process(monkeypatch, tmp_path) -> None:
    process = MagicMock()
    process.pid = 42
    process.returncode = None
    process.communicate = MagicMock(side_effect=asyncio.CancelledError)
    process.wait = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.shutil.which", lambda _: "/bin/claude"
    )
    monkeypatch.setattr(
        "openjarvis.engine.claude_cli.asyncio.create_subprocess_exec",
        MagicMock(return_value=asyncio.sleep(0, result=process)),
    )
    killpg = MagicMock()
    monkeypatch.setattr("openjarvis.engine.claude_cli.os.killpg", killpg)
    monkeypatch.setenv("OPENJARVIS_CLAUDE_CWD", str(tmp_path))
    monkeypatch.setattr(ClaudeCLIEngine, "_subscription_authenticated", lambda _: True)

    with pytest.raises(asyncio.CancelledError):
        async for _ in ClaudeCLIEngine().stream([], model="claude/default"):
            pass

    killpg.assert_called_once_with(42, 9)
