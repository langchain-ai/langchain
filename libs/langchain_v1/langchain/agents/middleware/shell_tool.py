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
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from langchain_core.messages import ToolMessage
from langchain_core.runnables import run_in_executor
from langchain_core.tools.base import ToolException
from langgraph.channels.untracked_value import UntrackedValue
from pydantic import BaseModel, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import NotRequired, override

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
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import ToolRuntime, tool

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langgraph.runtime import Runtime


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
SHELL_TOOL_NAME = "shell"


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
    finalizer: weakref.finalize = field(init=False, repr=False)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        self.finalizer = weakref.finalize(
            self,
            _cleanup_resources,
            self.session,
            self.tempdir,
            self.policy.termination_timeout,
        )


class ShellToolState(AgentState[ResponseT]):
    """Agent state extension for tracking shell session resources.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

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
        """Start the shell subprocess and reader threads.

        Raises:
            RuntimeError: If the shell session pipes cannot be initialized.
        """
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
            try:
                self._stdin.write(payload)
                self._stdin.write(f"printf '{marker} %s\\n' $?\n")
                self._stdin.flush()
            except (BrokenPipeError, OSError):
                # The shell exited before we could write the marker command.
                # This happens when commands like 'exit 1' terminate the shell.
                return self._collect_output_after_exit(deadline)

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
                # Drain any remaining stderr that may have arrived concurrently.
                # The stderr reader thread runs independently, so output might
                # still be in flight when the stdout marker arrives.
                self._drain_remaining_stderr(collected, deadline)
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

    def _collect_output_after_exit(self, deadline: float) -> CommandExecutionResult:
        """Collect output after the shell exited unexpectedly.

        Called when a `BrokenPipeError` occurs while writing to stdin, indicating the
        shell process terminated (e.g., due to an 'exit' command).

        Args:
            deadline: Absolute time by which collection must complete.

        Returns:
            `CommandExecutionResult` with collected output and the process exit code.
        """
        collected: list[str] = []
        total_lines = 0
        total_bytes = 0
        truncated_by_lines = False
        truncated_by_bytes = False

        # Give reader threads a brief moment to enqueue any remaining output.
        drain_timeout = 0.1
        drain_deadline = min(time.monotonic() + drain_timeout, deadline)

        while True:
            remaining = drain_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                source, data = self._queue.get(timeout=remaining)
            except queue.Empty:
                break

            if data is None:
                # EOF marker from a reader thread; continue draining.
                continue

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

        # Get exit code from the terminated process.
        exit_code: int | None = None
        if self._process:
            exit_code = self._process.poll()

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

    def _drain_remaining_stderr(
        self, collected: list[str], deadline: float, drain_timeout: float = 0.05
    ) -> None:
        """Drain any stderr output that arrived concurrently with the done marker.

        The stdout and stderr reader threads run independently. When a command writes to
        stderr just before exiting, the stderr output may still be in transit when the
        done marker arrives on stdout. This method briefly polls the queue to capture
        such output.

        Args:
            collected: The list to append collected stderr lines to.
            deadline: The original command deadline (used as an upper bound).
            drain_timeout: Maximum time to wait for additional stderr output.
        """
        drain_deadline = min(time.monotonic() + drain_timeout, deadline)
        while True:
            remaining = drain_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                source, data = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if data is None or source != "stderr":
                continue
            stripped = data.rstrip("\n")
            collected.append(f"[stderr] {stripped}")
            if data.endswith("\n"):
                collected.append("\n")

    @staticmethod
    def _safe_int(value: str) -> int | None:
        with contextlib.suppress(ValueError):
            return int(value)
        return None


class _ShellToolInput(BaseModel):
    """Input schema for the persistent shell tool."""

    command: str | None = None
    """The shell command to execute."""

    restart: bool | None = None
    """Whether to restart the shell session."""

    runtime: Annotated[Any, SkipJsonSchema()] = None
    """The runtime for the shell tool.

    Included as a workaround at the moment bc args_schema doesn't work with
    injected ToolRuntime.
    """

    @model_validator(mode="after")
    def validate_payload(self) -> _ShellToolInput:
        if self.command is None and not self.restart:
            msg = "Shell tool requires either 'command' or 'restart'."
            raise ValueError(msg)
        if self.command is not None and self.restart:
            msg = "Specify only one of 'command' or 'restart'."
            raise ValueError(msg)
        return self


class ShellToolMiddleware(AgentMiddleware[ShellToolState[ResponseT], ContextT, ResponseT]):
    """Middleware that registers a persistent shell tool for agents.

    The middleware exposes a single long-lived shell session. Use the execution policy
    to match your deployment's security posture:

    * `HostExecutionPolicy` – full host access; best for trusted environments where the
        agent already runs inside a container or VM that provides isolation.
    * `CodexSandboxExecutionPolicy` – reuses the Codex CLI sandbox for additional
        syscall/filesystem restrictions when the CLI is available.
    * `DockerExecutionPolicy` – launches a separate Docker container for each agent run,
        providing harder isolation, optional read-only root filesystems, and user
        remapping.

    When no policy is provided the middleware defaults to `HostExecutionPolicy`.
    """

    state_schema = ShellToolState  # type: ignore[assignment]

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        startup_commands: tuple[str, ...] | list[str] | str | None = None,
        shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
        execution_policy: BaseExecutionPolicy | None = None,
        redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None,
        tool_description: str | None = None,
        tool_name: str = SHELL_TOOL_NAME,
        shell_command: Sequence[str] | str | None = None,
        env: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize an instance of `ShellToolMiddleware`.

        Args:
            workspace_root: Base directory for the shell session.

                If omitted, a temporary directory is created when the agent starts and
                removed when it ends.
            startup_commands: Optional commands executed sequentially after the session
                starts.
            shutdown_commands: Optional commands executed before the session shuts down.
            execution_policy: Execution policy controlling timeouts, output limits, and
                resource configuration.

                Defaults to `HostExecutionPolicy` for native execution.
            redaction_rules: Optional redaction rules to sanitize command output before
                returning it to the model.

                !!! warning
                    Redaction rules are applied post execution and do not prevent
                    exfiltration of secrets or sensitive data when using
                    `HostExecutionPolicy`.

            tool_description: Optional override for the registered shell tool
                description.
            tool_name: Name for the registered shell tool.

                Defaults to `"shell"`.
            shell_command: Optional shell executable (string) or argument sequence used
                to launch the persistent session.

                Defaults to an implementation-defined bash command.
            env: Optional environment variables to supply to the shell session.

                Values are coerced to strings before command execution. If omitted, the
                session inherits the parent process environment.
        """
        super().__init__()
        self._workspace_root = Path(workspace_root) if workspace_root else None
        self._tool_name = tool_name
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

        # Create a proper tool that executes directly (no interception needed)
        description = tool_description or DEFAULT_TOOL_DESCRIPTION

        @tool(self._tool_name, args_schema=_ShellToolInput, description=description)
        def shell_tool(
            *,
            runtime: ToolRuntime[None, ShellToolState],
            command: str | None = None,
            restart: bool = False,
        ) -> ToolMessage | str:
            resources = self._get_or_create_resources(runtime.state)
            return self._run_shell_tool(
                resources,
                {"command": command, "restart": restart},
                tool_call_id=runtime.tool_call_id,
            )

        self._shell_tool = shell_tool
        self.tools = [self._shell_tool]

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
                msg = "Environment variable names must be strings."  # type: ignore[unreachable]
                raise TypeError(msg)
            normalized[key] = str(value)
        return normalized

    @override
    def before_agent(
        self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Start the shell session and run startup commands.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Shell session resources to be stored in the agent state.
        """
        resources = self._get_or_create_resources(state)
        return {"shell_session_resources": resources}

    async def abefore_agent(
        self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async start the shell session and run startup commands.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Shell session resources to be stored in the agent state.
        """
        return await run_in_executor(None, self.before_agent, state, runtime)

    @override
    def after_agent(self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]) -> None:
        """Run shutdown commands and release resources when an agent completes."""
        resources = state.get("shell_session_resources")
        if not isinstance(resources, _SessionResources):
            # Resources were never created, nothing to clean up
            return
        try:
            self._run_shutdown_commands(resources.session)
        finally:
            resources.finalizer()

    async def aafter_agent(
        self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]
    ) -> None:
        """Async run shutdown commands and release resources when an agent completes."""
        return self.after_agent(state, runtime)

    def _get_or_create_resources(self, state: ShellToolState[ResponseT]) -> _SessionResources:
        """Get existing resources from state or create new ones if they don't exist.

        This method enables resumability by checking if resources already exist in the state
        (e.g., after an interrupt), and only creating new resources if they're not present.

        Args:
            state: The agent state which may contain shell session resources.

        Returns:
            Session resources, either retrieved from state or newly created.
        """
        resources = state.get("shell_session_resources")
        if isinstance(resources, _SessionResources):
            return resources

        new_resources = self._create_resources()
        # Cast needed to make state dict-like for mutation
        cast("dict[str, Any]", state)["shell_session_resources"] = new_resources
        return new_resources

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
            if result.timed_out or (result.exit_code not in {0, None}):
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
                elif result.exit_code not in {0, None}:
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

        if result.exit_code not in {0, None}:
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
            name=self._tool_name,
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
