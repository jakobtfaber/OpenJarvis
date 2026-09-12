"""Codex CLI inference backend using saved ChatGPT authentication."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import EngineConnectionError, InferenceEngine

# ``codex exec`` has no flag that removes its tools, and neither a read-only
# sandbox nor a scrubbed child environment stops it reading files the user can
# read. Nothing observed after the child starts can undo such a read, so the
# only honest text-only enforcement is to refuse before launching one.
_TEXT_ONLY_UNSUPPORTED = (
    "The Codex CLI engine cannot serve a text-only turn: codex exec has no flag "
    "that removes its tools. Opt in explicitly with "
    "OPENJARVIS_FILESYSTEM_MODE=read-write to allow a tool-using turn scoped to "
    "OPENJARVIS_CODEX_CWD, or use the claude_cli engine, whose --restricted mode "
    "enforces text-only inside the CLI."
)
# Stripped from the child environment so a ChatGPT-subscription turn can never
# fall back to API-key billing or a redirected endpoint. This is a billing and
# endpoint guard, not a secret boundary: the child still inherits the rest of
# this process' environment and the caller's filesystem permissions.
_PROVIDER_ENV = frozenset(
    {
        "CODEX_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
    }
)
# How long a killed child is given to be reaped. The kill signal has already
# been delivered; this only bounds the wait for it to take effect.
_REAP_TIMEOUT_SECONDS = 5.0


def _parse_events(output: str) -> tuple[str, Dict[str, int]]:
    content = ""
    usage: Dict[str, int] = {}
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                content = item.get("text", "")
        elif event.get("type") == "turn.completed":
            raw_usage = event.get("usage", {})
            prompt_tokens = raw_usage.get("input_tokens", 0)
            completion_tokens = raw_usage.get("output_tokens", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            if "cached_input_tokens" in raw_usage:
                usage["cached_input_tokens"] = raw_usage["cached_input_tokens"]
        elif event.get("type") in {"error", "turn.failed"}:
            raise EngineConnectionError("Codex CLI turn failed")
    if not content:
        raise EngineConnectionError("Codex CLI returned no assistant message")
    return content, usage


@EngineRegistry.register("codex_cli")
class CodexCLIEngine(InferenceEngine):
    """Run inference through the subscription-authenticated Codex CLI."""

    engine_id = "codex_cli"
    is_cloud = True

    def __init__(self, *, timeout: float | None = None) -> None:
        self._binary = shutil.which("codex")
        self._timeout = timeout or float(
            os.environ.get("OPENJARVIS_CODEX_TIMEOUT", "120")
        )
        self._cwd = Path(
            os.environ.get(
                "OPENJARVIS_CODEX_CWD", os.environ.get("OPENJARVIS_HOME", Path.home())
            )
        ).expanduser()
        self._filesystem_enabled = (
            os.environ.get("OPENJARVIS_FILESYSTEM_MODE") == "read-write"
        )
        self._github_enabled = os.environ.get("OPENJARVIS_GITHUB_MODE") == "read-write"

    def _prompt(self, messages: Sequence[Message]) -> str:
        transcript = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        role = (
            f"You may read and modify files only inside {self._cwd}. "
            "Use local tools when needed, then return only the assistant reply."
        )
        if self._github_enabled:
            role += " Git and the authenticated GitHub CLI are available."
        return f"Act only as a text inference backend. {role}\n" + json.dumps(
            transcript, ensure_ascii=False
        )

    def _command(self, model: str) -> list[str]:
        # Fail closed before a child can exist: every caller that starts one
        # goes through here first.
        if not self._filesystem_enabled:
            raise EngineConnectionError(_TEXT_ONLY_UNSUPPORTED)
        if not self._binary:
            raise EngineConnectionError("Codex CLI is not installed")
        if not self._cwd.is_dir():
            raise EngineConnectionError(
                f"Codex working directory does not exist: {self._cwd}"
            )
        command = [
            self._binary,
            "exec",
            "--ephemeral",
            "--json",
            "--sandbox",
            "workspace-write",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-C",
            str(self._cwd),
        ]
        actual_model = model.removeprefix("codex/")
        if actual_model and actual_model != "default":
            command.extend(["--model", actual_model])
        command.append("-")
        return command

    def _environment(self) -> dict[str, str]:
        return {
            key: value for key, value in os.environ.items() if key not in _PROVIDER_ENV
        }

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        command = self._command(model)
        process = subprocess.Popen(
            command,
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
            raise EngineConnectionError("Codex CLI timed out") from exc
        if process.returncode:
            raise EngineConnectionError(
                f"Codex CLI failed with exit code {process.returncode}"
            )
        content, usage = _parse_events(stdout)
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
        command = self._command(model)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            env=self._environment(),
            cwd=self._cwd,
        )
        # One deadline for the whole turn — writing the prompt, every line of
        # output and the final exit. A per-read timeout lets a child that keeps
        # printing run for as long as it likes.
        deadline = monotonic() + self._timeout

        def remaining() -> float:
            return deadline - monotonic()

        try:
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(self._prompt(messages).encode())
            await asyncio.wait_for(process.stdin.drain(), remaining())
            process.stdin.close()
            while line := await asyncio.wait_for(
                process.stdout.readline(), remaining()
            ):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message" and item.get("text"):
                        yield item["text"]
                elif event.get("type") in {"error", "turn.failed"}:
                    raise EngineConnectionError("Codex CLI turn failed")
            if await asyncio.wait_for(process.wait(), remaining()):
                raise EngineConnectionError(
                    f"Codex CLI failed with exit code {process.returncode}"
                )
        except asyncio.TimeoutError as exc:
            raise EngineConnectionError("Codex CLI timed out") from exc
        finally:
            # Reached by every exit: a failed turn, a parse error, cancellation,
            # and the consumer closing the generator at a yield. Anything that
            # leaves the child running leaves it holding the workspace.
            await self._reap(process)

    @staticmethod
    async def _reap(process: Any) -> None:
        """Kill and collect the child unless it has already exited."""
        if process.returncode is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            return
        try:
            await asyncio.wait_for(process.wait(), _REAP_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Never let the cleanup replace the error that ended the turn.
            pass

    def list_models(self) -> List[str]:
        return ["codex/default"] if self.health() else []

    def health(self) -> bool:
        # Without the filesystem opt-in this engine cannot serve a turn at all,
        # so discovery must not advertise it as usable.
        if not self._filesystem_enabled:
            return False
        if not self._binary or not self._cwd.is_dir():
            return False
        try:
            result = subprocess.run(
                [self._binary, "login", "status"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                env=self._environment(),
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def can_serve(self, model: str) -> bool:
        return model == "default" or model.startswith("codex/")


__all__ = ["CodexCLIEngine"]
