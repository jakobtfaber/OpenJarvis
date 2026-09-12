"""Claude Code inference backend using saved Claude subscription authentication."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import stat
import subprocess
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Dict, List

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import EngineConnectionError, InferenceEngine

_TEXT_ONLY_SYSTEM_PROMPT = (
    "Act only as a text inference backend. Do not inspect files, run commands, "
    "or use tools. Return only the assistant reply to the JSON conversation "
    "supplied as user input."
)
_PROVIDER_ENV = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
}


def _parse_result(output: str) -> tuple[str, Dict[str, int]]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise EngineConnectionError("Claude CLI returned invalid JSON") from exc
    if payload.get("is_error") or payload.get("subtype") != "success":
        raise EngineConnectionError("Claude CLI turn failed")
    content = payload.get("result")
    if not isinstance(content, str) or not content:
        raise EngineConnectionError("Claude CLI returned no assistant message")
    raw_usage = payload.get("usage") or {}
    prompt_tokens = sum(
        int(raw_usage.get(key, 0) or 0)
        for key in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )
    completion_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if raw_usage.get("cache_read_input_tokens") is not None:
        usage["cached_input_tokens"] = int(
            raw_usage.get("cache_read_input_tokens", 0) or 0
        )
    return content, usage


@EngineRegistry.register("claude_cli")
class ClaudeCLIEngine(InferenceEngine):
    """Run inference through subscription-authenticated Claude Code."""

    engine_id = "claude_cli"
    is_cloud = True

    def __init__(self, *, timeout: float | None = None) -> None:
        self._binary = shutil.which("claude")
        self._timeout = timeout or float(
            os.environ.get("OPENJARVIS_CLAUDE_TIMEOUT", "120")
        )
        self._cwd = Path(
            os.environ.get(
                "OPENJARVIS_CLAUDE_CWD",
                os.environ.get("OPENJARVIS_HOME", Path.home()),
            )
        ).expanduser()
        self._filesystem_enabled = (
            os.environ.get("OPENJARVIS_FILESYSTEM_MODE") == "read-write"
        )
        self._github_enabled = os.environ.get("OPENJARVIS_GITHUB_MODE") == "read-write"

    @staticmethod
    def _prompt(messages: Sequence[Message]) -> str:
        transcript = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        return json.dumps(transcript, ensure_ascii=False)

    def _command(self, model: str) -> list[str]:
        if not self._binary:
            raise EngineConnectionError("Claude CLI is not installed")
        if not self._cwd.is_dir():
            raise EngineConnectionError(
                f"Claude working directory does not exist: {self._cwd}"
            )
        enabled_tools: list[str] = []
        if self._filesystem_enabled:
            enabled_tools.extend(["Read", "Write", "Edit", "Glob", "Grep"])
            if self._github_enabled:
                enabled_tools.append("Bash")
        tools = ",".join(enabled_tools)
        permission_mode = "acceptEdits" if self._filesystem_enabled else "plan"
        max_turns = "8" if self._filesystem_enabled else "1"
        system_prompt = (
            f"Act only as a text inference backend. You may read and modify files "
            f"only inside {self._cwd}. Use the available file tools when needed, "
            "then return only the assistant reply."
            if self._filesystem_enabled
            else _TEXT_ONLY_SYSTEM_PROMPT
        )
        command = [
            self._binary,
            "--print",
            "--output-format",
            "json",
            "--safe-mode",
            # --restricted drops the command-running tools unless --tools names
            # them and confines the file tools to the working directories, so
            # the boundaries below are enforced by the CLI, not just requested
            # in the system prompt.
            "--restricted",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--tools",
            tools,
            "--permission-mode",
            permission_mode,
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--prompt-suggestions",
            "false",
            "--max-turns",
            max_turns,
            "--system-prompt",
            system_prompt,
        ]
        if self._filesystem_enabled:
            command.extend(["--add-dir", str(self._cwd)])
            if self._github_enabled:
                command.extend(["--allowed-tools", "Bash(gh *)", "Bash(git *)"])
        actual_model = model.removeprefix("claude/")
        if actual_model and actual_model != "default":
            command.extend(["--model", actual_model])
        return command

    def _environment(self) -> dict[str, str]:
        environment = {
            key: value for key, value in os.environ.items() if key not in _PROVIDER_ENV
        }
        environment.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        token_path = os.environ.get("OPENJARVIS_CLAUDE_OAUTH_TOKEN_FILE")
        if token_path:
            path = Path(token_path).expanduser()
            try:
                metadata = path.stat()
                if (
                    metadata.st_uid == os.getuid()
                    and stat.S_IMODE(metadata.st_mode) == 0o600
                ):
                    token = path.read_text().strip()
                    if token:
                        environment["CLAUDE_CODE_OAUTH_TOKEN"] = token
            except OSError:
                pass
        return environment

    def _subscription_authenticated(self) -> bool:
        if not self._binary or not self._cwd.is_dir():
            return False
        environment = self._environment()
        try:
            result = subprocess.run(
                [self._binary, "auth", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                env=environment,
            )
            status = json.loads(result.stdout) if result.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return False
        if not status.get("loggedIn"):
            return False
        if status.get("authMethod") == "oauth_token":
            return bool(environment.get("CLAUDE_CODE_OAUTH_TOKEN"))
        return bool(
            status.get("authMethod") == "claude.ai" and status.get("subscriptionType")
        )

    def _require_subscription(self) -> None:
        if not self._subscription_authenticated():
            raise EngineConnectionError("Claude subscription login is unavailable")

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._require_subscription()
        process = subprocess.Popen(
            self._command(model),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=os.name != "nt",
            env=self._environment(),
            cwd=self._cwd,
        )
        try:
            stdout, _ = process.communicate(
                self._prompt(messages), timeout=self._timeout
            )
        except subprocess.TimeoutExpired as exc:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise EngineConnectionError("Claude CLI timed out") from exc
        if process.returncode:
            raise EngineConnectionError(
                f"Claude CLI failed with exit code {process.returncode}"
            )
        content, usage = _parse_result(stdout)
        return {
            "content": content,
            "usage": usage,
            "model": model,
            "finish_reason": "stop",
        }

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self._require_subscription()
        process = await asyncio.create_subprocess_exec(
            *self._command(model),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            env=self._environment(),
            cwd=self._cwd,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(self._prompt(messages).encode()), self._timeout
            )
        except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
            if process.returncode is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                await process.wait()
            if isinstance(exc, asyncio.TimeoutError):
                raise EngineConnectionError("Claude CLI timed out") from exc
            raise
        if process.returncode:
            raise EngineConnectionError(
                f"Claude CLI failed with exit code {process.returncode}"
            )
        content, _ = _parse_result(stdout.decode())
        yield content

    def list_models(self) -> List[str]:
        return ["claude/default"] if self._binary and self._cwd.is_dir() else []

    def health(self) -> bool:
        return self._subscription_authenticated()

    def can_serve(self, model: str) -> bool:
        return model == "default" or model.startswith("claude/")


__all__ = ["ClaudeCLIEngine"]
