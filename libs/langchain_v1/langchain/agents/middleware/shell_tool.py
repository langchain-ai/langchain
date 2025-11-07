"""Middleware that exposes a persistent shell tool to agents."""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import typing
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools.base import BaseTool, ToolException
from langgraph.channels.untracked_value import UntrackedValue
from pydantic import BaseModel, model_validator
from typing_extensions import NotRequired

from langchain.agents.middleware._execution import (
    SHELL_TEMP_PREFIX,
    BaseExecutionPolicy,
    CodexSandboxExecutionPolicy,
    DockerExecutionPolicy,
    HostExecutionPolicy,
)
from langchain.agents.middleware._redaction import (
    PIIDetectionError,
    PIIMatch,
    RedactionRule,
    ResolvedRedactionRule,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, PrivateStateAttr

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest

LOGGER = logging.getLogger(__name__)
_DONE_MARKER_PREFIX = "__LC_SHELL_DONE__"

DEFAULT_TOOL_DESCRIPTION = (
    "Execute a shell command inside a persistent session. Before running a command, "
    "confirm the working directory is correct (e.g., inspect with `ls` or `pwd`) and ensure "
    "any parent directories exist. Prefer absolute paths and quote paths containing spaces, "
    'such as `cd "/path/with spaces"`. Chain multiple commands with `&&` or `;` instead of '
    "embedding newlines. Avoid unnecessary `cd` usage unless explicitly required so the "
    "session remains stable. Outputs may be truncated when they become very large, and long "
    "running commands will be terminated once their configured timeout elapses."
)


def _cleanup_resources(
    session: ShellSession, tempdir: tempfile.TemporaryDirectory[str] | None, timeout: float
) -> None:
    with contextlib.suppress(Exception):
        session.stop(timeout)
    if tempdir is not None:
        with contextlib.suppress(Exception):
            tempdir.cleanup()


@dataclass
class _SessionResources:
    """Container for per-run shell resources."""

    session: ShellSession
    tempdir: tempfile.TemporaryDirectory[str] | None
    policy: BaseExecutionPolicy
    _finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._finalizer = weakref.finalize(
            self,
            _cleanup_resources,
            self.session,
            self.tempdir,
            self.policy.termination_timeout,
        )


class ShellToolState(AgentState):
    """Agent state extension for tracking shell session resources."""

    shell_session_resources: NotRequired[
        Annotated[_SessionResources | None, UntrackedValue, PrivateStateAttr]
    ]


@dataclass(frozen=True)
class CommandExecutionResult:
    """Structured result from command execution."""

    output: str
    exit_code: int | None
    timed_out: bool
    truncated_by_lines: bool
    truncated_by_bytes: bool
    total_lines: int
    total_bytes: int


class ShellSession:
    """Persistent shell session that supports sequential command execution."""

    def __init__(
        self,
        workspace: Path,
        policy: BaseExecutionPolicy,
        command: tuple[str, ...],
        environment: Mapping[str, str],
    ) -> None:
        self._workspace = workspace
        self._policy = policy
        self._command = command
        self._environment = dict(environment)
        self._process: subprocess.Popen[str] | None = None
        self._stdin: Any = None
        self._queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._lock = threading.Lock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._terminated = False

    def start(self) -> None:
        """Start the shell subprocess and reader threads."""
        if self._process and self._process.poll() is None:
            return

        self._process = self._policy.spawn(
            workspace=self._workspace,
            env=self._environment,
            command=self._command,
        )
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            msg = "Failed to initialize shell session pipes."
            raise RuntimeError(msg)

        self._stdin = self._process.stdin
        self._terminated = False
        self._queue = queue.Queue()

        self._stdout_thread = threading.Thread(
            target=self._enqueue_stream,
            args=(self._process.stdout, "stdout"),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._enqueue_stream,
            args=(self._process.stderr, "stderr"),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def restart(self) -> None:
        """Restart the shell process."""
        self.stop(self._policy.termination_timeout)
        self.start()

    def stop(self, timeout: float) -> None:
        """Stop the shell subprocess."""
        if not self._process:
            return

        if self._process.poll() is None and not self._terminated:
            try:
                self._stdin.write("exit\n")
                self._stdin.flush()
            except (BrokenPipeError, OSError):
                LOGGER.debug(
                    "Failed to write exit command; terminating shell session.",
                    exc_info=True,
                )

        try:
            if self._process.wait(timeout=timeout) is None:
                self._kill_process()
        except subprocess.TimeoutExpired:
            self._kill_process()
        finally:
            self._terminated = True
            with contextlib.suppress(Exception):
                self._stdin.close()
            self._process = None

    def execute(self, command: str, *, timeout: float) -> CommandExecutionResult:
        """Execute a command in the persistent shell."""
        if not self._process or self._process.poll() is not None:
            msg = "Shell session is not running."
            raise RuntimeError(msg)

        marker = f"{_DONE_MARKER_PREFIX}{uuid.uuid4().hex}"
        deadline = time.monotonic() + timeout

        with self._lock:
            self._drain_queue()
            payload = command if command.endswith("\n") else f"{command}\n"
            self._stdin.write(payload)
            self._stdin.write(f"printf '{marker} %s\\n' $?\n")
            self._stdin.flush()

            return self._collect_output(marker, deadline, timeout)

    def _collect_output(
        self,
        marker: str,
        deadline: float,
        timeout: float,
    ) -> CommandExecutionResult:
        collected: list[str] = []
        total_lines = 0
        total_bytes = 0
        truncated_by_lines = False
        truncated_by_bytes = False
        exit_code: int | None = None
        timed_out = False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                source, data = self._queue.get(timeout=remaining)
            except queue.Empty:
                timed_out = True
                break

            if data is None:
                continue

            if source == "stdout" and data.startswith(marker):
                _, _, status = data.partition(" ")
                exit_code = self._safe_int(status.strip())
                break

            total_lines += 1
            encoded = data.encode("utf-8", "replace")
            total_bytes += len(encoded)

            if total_lines > self._policy.max_output_lines:
                truncated_by_lines = True
                continue

            if (
                self._policy.max_output_bytes is not None
                and total_bytes > self._policy.max_output_bytes
            ):
                truncated_by_bytes = True
                continue

            if source == "stderr":
                stripped = data.rstrip("\n")
                collected.append(f"[stderr] {stripped}")
                if data.endswith("\n"):
                    collected.append("\n")
            else:
                collected.append(data)

        if timed_out:
            LOGGER.warning(
                "Command timed out after %.2f seconds; restarting shell session.",
                timeout,
            )
            self.restart()
            return CommandExecutionResult(
                output="",
                exit_code=None,
                timed_out=True,
                truncated_by_lines=truncated_by_lines,
                truncated_by_bytes=truncated_by_bytes,
                total_lines=total_lines,
                total_bytes=total_bytes,
            )

        output = "".join(collected)
        return CommandExecutionResult(
            output=output,
            exit_code=exit_code,
            timed_out=False,
            truncated_by_lines=truncated_by_lines,
            truncated_by_bytes=truncated_by_bytes,
            total_lines=total_lines,
            total_bytes=total_bytes,
        )

    def _kill_process(self) -> None:
        if not self._process:
            return

        if hasattr(os, "killpg"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
        else:  # pragma: no cover
            with contextlib.suppress(ProcessLookupError):
                self._process.kill()

    def _enqueue_stream(self, stream: Any, label: str) -> None:
        for line in iter(stream.readline, ""):
            self._queue.put((label, line))
        self._queue.put((label, None))

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _safe_int(value: str) -> int | None:
        with contextlib.suppress(ValueError):
            return int(value)
        return None


class _ShellToolInput(BaseModel):
    """Input schema for the persistent shell tool."""

    command: str | None = None
    restart: bool | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> _ShellToolInput:
        if self.command is None and not self.restart:
            msg = "Shell tool requires either 'command' or 'restart'."
            raise ValueError(msg)
        if self.command is not None and self.restart:
            msg = "Specify only one of 'command' or 'restart'."
            raise ValueError(msg)
        return self


class _PersistentShellTool(BaseTool):
    """Tool wrapper that relies on middleware interception for execution."""

    name: str = "shell"
    description: str = DEFAULT_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = _ShellToolInput

    def __init__(self, middleware: ShellToolMiddleware, description: str | None = None) -> None:
        super().__init__()
        self._middleware = middleware
        if description is not None:
            self.description = description

    def _run(self, **_: Any) -> Any:  # pragma: no cover - executed via middleware wrapper
        msg = "Persistent shell tool execution should be intercepted via middleware wrappers."
        raise RuntimeError(msg)


class ShellToolMiddleware(AgentMiddleware[ShellToolState, Any]):
    """Middleware that registers a persistent shell tool for agents.

    The middleware exposes a single long-lived shell session. Use the execution policy to
    match your deployment's security posture:

    * ``HostExecutionPolicy`` - full host access; best for trusted environments where the
      agent already runs inside a container or VM that provides isolation.
    * ``CodexSandboxExecutionPolicy`` - reuses the Codex CLI sandbox for additional
      syscall/filesystem restrictions when the CLI is available.
    * ``DockerExecutionPolicy`` - launches a separate Docker container for each agent run,
      providing harder isolation, optional read-only root filesystems, and user remapping.

    When no policy is provided the middleware defaults to ``HostExecutionPolicy``.
    """

    state_schema = ShellToolState

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        startup_commands: tuple[str, ...] | list[str] | str | None = None,
        shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
        execution_policy: BaseExecutionPolicy | None = None,
        redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None,
        tool_description: str | None = None,
        shell_command: Sequence[str] | str | None = None,
        env: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            workspace_root: Base directory for the shell session. If omitted, a temporary
                directory is created when the agent starts and removed when it ends.
            startup_commands: Optional commands executed sequentially after the session starts.
            shutdown_commands: Optional commands executed before the session shuts down.
            execution_policy: Execution policy controlling timeouts, output limits, and resource
                configuration. Defaults to :class:`HostExecutionPolicy` for native execution.
            redaction_rules: Optional redaction rules to sanitize command output before
                returning it to the model.
            tool_description: Optional override for the registered shell tool description.
            shell_command: Optional shell executable (string) or argument sequence used to
                launch the persistent session. Defaults to an implementation-defined bash command.
            env: Optional environment variables to supply to the shell session. Values are
                coerced to strings before command execution. If omitted, the session inherits the
                parent process environment.
        """
        super().__init__()
        self._workspace_root = Path(workspace_root) if workspace_root else None
        self._shell_command = self._normalize_shell_command(shell_command)
        self._environment = self._normalize_env(env)
        if execution_policy is not None:
            self._execution_policy = execution_policy
        else:
            self._execution_policy = HostExecutionPolicy()
        rules = redaction_rules or ()
        self._redaction_rules: tuple[ResolvedRedactionRule, ...] = tuple(
            rule.resolve() for rule in rules
        )
        self._startup_commands = self._normalize_commands(startup_commands)
        self._shutdown_commands = self._normalize_commands(shutdown_commands)

        description = tool_description or DEFAULT_TOOL_DESCRIPTION
        self._tool = _PersistentShellTool(self, description=description)
        self.tools = [self._tool]

    @staticmethod
    def _normalize_commands(
        commands: tuple[str, ...] | list[str] | str | None,
    ) -> tuple[str, ...]:
        if commands is None:
            return ()
        if isinstance(commands, str):
            return (commands,)
        return tuple(commands)

    @staticmethod
    def _normalize_shell_command(
        shell_command: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if shell_command is None:
            return ("/bin/bash",)
        normalized = (shell_command,) if isinstance(shell_command, str) else tuple(shell_command)
        if not normalized:
            msg = "Shell command must contain at least one argument."
            raise ValueError(msg)
        return normalized

    @staticmethod
    def _normalize_env(env: Mapping[str, Any] | None) -> dict[str, str] | None:
        if env is None:
            return None
        normalized: dict[str, str] = {}
        for key, value in env.items():
            if not isinstance(key, str):
                msg = "Environment variable names must be strings."
                raise TypeError(msg)
            normalized[key] = str(value)
        return normalized

    def before_agent(self, state: ShellToolState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Start the shell session and run startup commands."""
        resources = self._create_resources()
        return {"shell_session_resources": resources}

    async def abefore_agent(self, state: ShellToolState, runtime: Runtime) -> dict[str, Any] | None:
        """Async counterpart to `before_agent`."""
        return self.before_agent(state, runtime)

    def after_agent(self, state: ShellToolState, runtime: Runtime) -> None:  # noqa: ARG002
        """Run shutdown commands and release resources when an agent completes."""
        resources = self._ensure_resources(state)
        try:
            self._run_shutdown_commands(resources.session)
        finally:
            resources._finalizer()

    async def aafter_agent(self, state: ShellToolState, runtime: Runtime) -> None:
        """Async counterpart to `after_agent`."""
        return self.after_agent(state, runtime)

    def _ensure_resources(self, state: ShellToolState) -> _SessionResources:
        resources = state.get("shell_session_resources")
        if resources is not None and not isinstance(resources, _SessionResources):
            resources = None
        if resources is None:
            msg = (
                "Shell session resources are unavailable. Ensure `before_agent` ran successfully "
                "before invoking the shell tool."
            )
            raise ToolException(msg)
        return resources

    def _create_resources(self) -> _SessionResources:
        workspace = self._workspace_root
        tempdir: tempfile.TemporaryDirectory[str] | None = None
        if workspace is None:
            tempdir = tempfile.TemporaryDirectory(prefix=SHELL_TEMP_PREFIX)
            workspace_path = Path(tempdir.name)
        else:
            workspace_path = workspace
            workspace_path.mkdir(parents=True, exist_ok=True)

        session = ShellSession(
            workspace_path,
            self._execution_policy,
            self._shell_command,
            self._environment or {},
        )
        try:
            session.start()
            LOGGER.info("Started shell session in %s", workspace_path)
            self._run_startup_commands(session)
        except BaseException:
            LOGGER.exception("Starting shell session failed; cleaning up resources.")
            session.stop(self._execution_policy.termination_timeout)
            if tempdir is not None:
                tempdir.cleanup()
            raise

        return _SessionResources(session=session, tempdir=tempdir, policy=self._execution_policy)

    def _run_startup_commands(self, session: ShellSession) -> None:
        if not self._startup_commands:
            return
        for command in self._startup_commands:
            result = session.execute(command, timeout=self._execution_policy.startup_timeout)
            if result.timed_out or (result.exit_code not in (0, None)):
                msg = f"Startup command '{command}' failed with exit code {result.exit_code}"
                raise RuntimeError(msg)

    def _run_shutdown_commands(self, session: ShellSession) -> None:
        if not self._shutdown_commands:
            return
        for command in self._shutdown_commands:
            try:
                result = session.execute(command, timeout=self._execution_policy.command_timeout)
                if result.timed_out:
                    LOGGER.warning("Shutdown command '%s' timed out.", command)
                elif result.exit_code not in (0, None):
                    LOGGER.warning(
                        "Shutdown command '%s' exited with %s.", command, result.exit_code
                    )
            except (RuntimeError, ToolException, OSError) as exc:
                LOGGER.warning(
                    "Failed to run shutdown command '%s': %s", command, exc, exc_info=True
                )

    def _apply_redactions(self, content: str) -> tuple[str, dict[str, list[PIIMatch]]]:
        """Apply configured redaction rules to command output."""
        matches_by_type: dict[str, list[PIIMatch]] = {}
        updated = content
        for rule in self._redaction_rules:
            updated, matches = rule.apply(updated)
            if matches:
                matches_by_type.setdefault(rule.pii_type, []).extend(matches)
        return updated, matches_by_type

    def _run_shell_tool(
        self,
        resources: _SessionResources,
        payload: dict[str, Any],
        *,
        tool_call_id: str | None,
    ) -> Any:
        session = resources.session

        if payload.get("restart"):
            LOGGER.info("Restarting shell session on request.")
            try:
                session.restart()
                self._run_startup_commands(session)
            except BaseException as err:
                LOGGER.exception("Restarting shell session failed; session remains unavailable.")
                msg = "Failed to restart shell session."
                raise ToolException(msg) from err
            message = "Shell session restarted."
            return self._format_tool_message(message, tool_call_id, status="success")

        command = payload.get("command")
        if not command or not isinstance(command, str):
            msg = "Shell tool expects a 'command' string when restart is not requested."
            raise ToolException(msg)

        LOGGER.info("Executing shell command: %s", command)
        result = session.execute(command, timeout=self._execution_policy.command_timeout)

        if result.timed_out:
            timeout_seconds = self._execution_policy.command_timeout
            message = f"Error: Command timed out after {timeout_seconds:.1f} seconds."
            return self._format_tool_message(
                message,
                tool_call_id,
                status="error",
                artifact={
                    "timed_out": True,
                    "exit_code": None,
                },
            )

        try:
            sanitized_output, matches = self._apply_redactions(result.output)
        except PIIDetectionError as error:
            LOGGER.warning("Blocking command output due to detected %s.", error.pii_type)
            message = f"Output blocked: detected {error.pii_type}."
            return self._format_tool_message(
                message,
                tool_call_id,
                status="error",
                artifact={
                    "timed_out": False,
                    "exit_code": result.exit_code,
                    "matches": {error.pii_type: error.matches},
                },
            )

        sanitized_output = sanitized_output or "<no output>"
        if result.truncated_by_lines:
            sanitized_output = (
                f"{sanitized_output.rstrip()}\n\n"
                f"... Output truncated at {self._execution_policy.max_output_lines} lines "
                f"(observed {result.total_lines})."
            )
        if result.truncated_by_bytes and self._execution_policy.max_output_bytes is not None:
            sanitized_output = (
                f"{sanitized_output.rstrip()}\n\n"
                f"... Output truncated at {self._execution_policy.max_output_bytes} bytes "
                f"(observed {result.total_bytes})."
            )

        if result.exit_code not in (0, None):
            sanitized_output = f"{sanitized_output.rstrip()}\n\nExit code: {result.exit_code}"
            final_status: Literal["success", "error"] = "error"
        else:
            final_status = "success"

        artifact = {
            "timed_out": False,
            "exit_code": result.exit_code,
            "truncated_by_lines": result.truncated_by_lines,
            "truncated_by_bytes": result.truncated_by_bytes,
            "total_lines": result.total_lines,
            "total_bytes": result.total_bytes,
            "redaction_matches": matches,
        }

        return self._format_tool_message(
            sanitized_output,
            tool_call_id,
            status=final_status,
            artifact=artifact,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: typing.Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept local shell tool calls and execute them via the managed session."""
        if isinstance(request.tool, _PersistentShellTool):
            resources = self._ensure_resources(request.state)
            return self._run_shell_tool(
                resources,
                request.tool_call["args"],
                tool_call_id=request.tool_call.get("id"),
            )
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: typing.Callable[[ToolCallRequest], typing.Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async interception mirroring the synchronous tool handler."""
        if isinstance(request.tool, _PersistentShellTool):
            resources = self._ensure_resources(request.state)
            return self._run_shell_tool(
                resources,
                request.tool_call["args"],
                tool_call_id=request.tool_call.get("id"),
            )
        return await handler(request)

    def _format_tool_message(
        self,
        content: str,
        tool_call_id: str | None,
        *,
        status: Literal["success", "error"],
        artifact: dict[str, Any] | None = None,
    ) -> ToolMessage | str:
        artifact = artifact or {}
        if tool_call_id is None:
            return content
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=self._tool.name,
            status=status,
            artifact=artifact,
        )


__all__ = [
    "CodexSandboxExecutionPolicy",
    "DockerExecutionPolicy",
    "HostExecutionPolicy",
    "RedactionRule",
    "ShellToolMiddleware",
]
