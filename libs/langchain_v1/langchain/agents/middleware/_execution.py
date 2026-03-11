"""Execution policies for the persistent shell middleware."""

from __future__ import annotations

import abc
import json
import os
import shutil
import subprocess
import sys
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - optional dependency on POSIX platforms
    import resource

    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - non-POSIX systems
    _HAS_RESOURCE = False


SHELL_TEMP_PREFIX = "langchain-shell-"


def _launch_subprocess(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    preexec_fn: typing.Callable[[], None] | None,
    start_new_session: bool,
) -> subprocess.Popen[str]:
    """Launch a subprocess with standardised I/O and encoding settings.

    All execution policies delegate to this helper so that stream handling,
    encoding, and buffering are consistent across policy implementations.

    Args:
        command: The command and arguments to execute.
        env: Environment variables for the subprocess.
        cwd: Working directory for the subprocess.
        preexec_fn: Optional callable invoked in the child process before exec.
        start_new_session: Whether to start the process in a new session.

    Returns:
        The spawned `subprocess.Popen` instance.
    """
    return subprocess.Popen(  # noqa: S603
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        preexec_fn=preexec_fn,  # noqa: PLW1509
        start_new_session=start_new_session,
    )


if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


@dataclass
class BaseExecutionPolicy(abc.ABC):
    """Configuration contract for persistent shell sessions.

    Concrete subclasses encapsulate how a shell process is launched and constrained.

    Each policy documents its security guarantees and the operating environments in
    which it is appropriate. Use `HostExecutionPolicy` for trusted, same-host execution;
    `CodexSandboxExecutionPolicy` when the Codex CLI sandbox is available and you want
    additional syscall restrictions; and `DockerExecutionPolicy` for container-level
    isolation using Docker.
    """

    command_timeout: float = 30.0
    startup_timeout: float = 30.0
    termination_timeout: float = 10.0
    max_output_lines: int = 100
    max_output_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_output_lines <= 0:
            msg = "max_output_lines must be positive."
            raise ValueError(msg)

    @abc.abstractmethod
    def spawn(
        self,
        *,
        workspace: Path,
        env: Mapping[str, str],
        command: Sequence[str],
    ) -> subprocess.Popen[str]:
        """Launch the persistent shell process."""


@dataclass
class HostExecutionPolicy(BaseExecutionPolicy):
    """Run the shell directly on the host process.

    This policy is best suited for trusted or single-tenant environments (CI jobs,
    developer workstations, pre-sandboxed containers) where the agent must access the
    host filesystem and tooling without additional isolation. Enforces optional CPU and
    memory limits to prevent runaway commands but offers **no** filesystem or network
    sandboxing; commands can modify anything the process user can reach.

    On Linux platforms resource limits are applied with `resource.prlimit` after the
    shell starts. On macOS, where `prlimit` is unavailable, limits are set in a
    `preexec_fn` before `exec`. In both cases the shell runs in its own process group
    so timeouts can terminate the full subtree.
    """

    cpu_time_seconds: int | None = None
    memory_bytes: int | None = None
    create_process_group: bool = True

    _limits_requested: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cpu_time_seconds is not None and self.cpu_time_seconds <= 0:
            msg = "cpu_time_seconds must be positive if provided."
            raise ValueError(msg)
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            msg = "memory_bytes must be positive if provided."
            raise ValueError(msg)
        self._limits_requested = any(
            value is not None for value in (self.cpu_time_seconds, self.memory_bytes)
        )
        if self._limits_requested and not _HAS_RESOURCE:
            msg = (
                "HostExecutionPolicy cpu/memory limits require the Python 'resource' module. "
                "Either remove the limits or run on a POSIX platform."
            )
            raise RuntimeError(msg)

    def spawn(
        self,
        *,
        workspace: Path,
        env: Mapping[str, str],
        command: Sequence[str],
    ) -> subprocess.Popen[str]:
        process = _launch_subprocess(
            list(command),
            env=env,
            cwd=workspace,
            preexec_fn=self._create_preexec_fn(),
            start_new_session=self.create_process_group,
        )
        self._apply_post_spawn_limits(process)
        return process

    def _create_preexec_fn(self) -> typing.Callable[[], None] | None:
        """Create a pre-exec callback that sets resource limits in the child process.

        On platforms where `prlimit` is available (Linux), limits are applied
        after spawn instead, so this returns `None`. On other POSIX platforms
        (e.g. macOS) the returned callback configures limits via
        `resource.setrlimit` before exec.

        Returns:
            A callable that sets CPU and memory limits, or `None` if limits
            will be applied post-spawn.
        """
        if not self._limits_requested or self._can_use_prlimit():
            return None

        def _configure() -> None:  # pragma: no cover - depends on OS
            if self.cpu_time_seconds is not None:
                limit = (self.cpu_time_seconds, self.cpu_time_seconds)
                resource.setrlimit(resource.RLIMIT_CPU, limit)
            if self.memory_bytes is not None:
                limit = (self.memory_bytes, self.memory_bytes)
                if hasattr(resource, "RLIMIT_AS"):
                    resource.setrlimit(resource.RLIMIT_AS, limit)
                elif hasattr(resource, "RLIMIT_DATA"):
                    resource.setrlimit(resource.RLIMIT_DATA, limit)

        return _configure

    def _apply_post_spawn_limits(self, process: subprocess.Popen[str]) -> None:
        """Apply CPU and memory limits to an already-running process via `prlimit`.

        This is the Linux-specific path. On platforms where `prlimit` is
        unavailable, limits are set in the pre-exec callback instead.

        Args:
            process: The running subprocess to constrain.

        Raises:
            RuntimeError: If `prlimit` fails (e.g. insufficient permissions).
        """
        if not self._limits_requested or not self._can_use_prlimit():
            return
        if not _HAS_RESOURCE:  # pragma: no cover - defensive
            return
        pid = process.pid
        try:
            prlimit = typing.cast("typing.Any", resource).prlimit
            if self.cpu_time_seconds is not None:
                prlimit(pid, resource.RLIMIT_CPU, (self.cpu_time_seconds, self.cpu_time_seconds))
            if self.memory_bytes is not None:
                limit = (self.memory_bytes, self.memory_bytes)
                if hasattr(resource, "RLIMIT_AS"):
                    prlimit(pid, resource.RLIMIT_AS, limit)
                elif hasattr(resource, "RLIMIT_DATA"):
                    prlimit(pid, resource.RLIMIT_DATA, limit)
        except OSError as exc:  # pragma: no cover - depends on platform support
            msg = "Failed to apply resource limits via prlimit."
            raise RuntimeError(msg) from exc

    @staticmethod
    def _can_use_prlimit() -> bool:
        """Check whether `resource.prlimit` is available on this platform.

        Returns:
            `True` on Linux when the `resource` module exposes `prlimit`.
        """
        return _HAS_RESOURCE and hasattr(resource, "prlimit") and sys.platform.startswith("linux")


@dataclass
class CodexSandboxExecutionPolicy(BaseExecutionPolicy):
    """Launch the shell through the Codex CLI sandbox.

    Ideal when you have the Codex CLI installed and want the additional syscall and
    filesystem restrictions provided by Anthropic's Seatbelt (macOS) or Landlock/seccomp
    (Linux) profiles. Commands still run on the host, but within the sandbox requested by
    the CLI. If the Codex binary is unavailable or the runtime lacks the required
    kernel features (e.g., Landlock inside some containers), process startup fails with a
    `RuntimeError`.

    Configure sandbox behavior via `config_overrides` to align with your Codex CLI
    profile. This policy does not add its own resource limits; combine it with
    host-level guards (cgroups, container resource limits) as needed.
    """

    binary: str = "codex"
    platform: typing.Literal["auto", "macos", "linux"] = "auto"
    config_overrides: Mapping[str, typing.Any] = field(default_factory=dict)

    def spawn(
        self,
        *,
        workspace: Path,
        env: Mapping[str, str],
        command: Sequence[str],
    ) -> subprocess.Popen[str]:
        full_command = self._build_command(command)
        return _launch_subprocess(
            full_command,
            env=env,
            cwd=workspace,
            preexec_fn=None,
            start_new_session=False,
        )

    def _build_command(self, command: Sequence[str]) -> list[str]:
        """Assemble the full Codex CLI sandbox invocation.

        Args:
            command: The user command to run inside the sandbox.

        Returns:
            The complete argument list for `subprocess.Popen`.
        """
        binary = self._resolve_binary()
        platform_arg = self._determine_platform()
        full_command: list[str] = [binary, "sandbox", platform_arg]
        for key, value in sorted(dict(self.config_overrides).items()):
            full_command.extend(["-c", f"{key}={self._format_override(value)}"])
        full_command.append("--")
        full_command.extend(command)
        return full_command

    def _resolve_binary(self) -> str:
        """Locate the Codex CLI binary on `PATH`.

        Returns:
            The absolute path to the resolved binary.

        Raises:
            RuntimeError: If the binary is not found.
        """
        path = shutil.which(self.binary)
        if path is None:
            msg = (
                "Codex sandbox policy requires the '%s' CLI to be installed and available on PATH."
            )
            raise RuntimeError(msg % self.binary)
        return path

    def _determine_platform(self) -> str:
        """Resolve the sandbox platform identifier.

        When `platform` is set to ``"auto"``, the current OS is inspected.

        Returns:
            ``"linux"`` or ``"macos"``.

        Raises:
            RuntimeError: If the platform cannot be determined automatically.
        """
        if self.platform != "auto":
            return self.platform
        if sys.platform.startswith("linux"):
            return "linux"
        if sys.platform == "darwin":  # type: ignore[unreachable, unused-ignore]
            return "macos"
        msg = (  # type: ignore[unreachable, unused-ignore]
            "Codex sandbox policy could not determine a supported platform; "
            "set 'platform' explicitly."
        )
        raise RuntimeError(msg)

    @staticmethod
    def _format_override(value: typing.Any) -> str:
        """Serialize a config override value to a string for the CLI.

        Args:
            value: The override value to serialize.

        Returns:
            A JSON string when the value is JSON-serializable, otherwise
            the `str` representation.
        """
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)


@dataclass
class DockerExecutionPolicy(BaseExecutionPolicy):
    """Run the shell inside a dedicated Docker container.

    Choose this policy when commands originate from untrusted users or you require
    strong isolation between sessions. By default the workspace is bind-mounted only
    when it refers to an existing non-temporary directory; ephemeral sessions run
    without a mount to minimise host exposure. The container's network namespace is
    disabled by default (`--network none`) and you can enable further hardening via
    `read_only_rootfs` and `user`.

    The security guarantees depend on your Docker daemon configuration. Run the agent on
    a host where Docker is locked down (rootless mode, AppArmor/SELinux, etc.) and
    review any additional volumes or capabilities passed through ``extra_run_args``. The
    default image is `python:3.12-alpine3.19`; supply a custom image if you need
    preinstalled tooling.
    """

    binary: str = "docker"
    image: str = "python:3.12-alpine3.19"
    remove_container_on_exit: bool = True
    network_enabled: bool = False
    extra_run_args: Sequence[str] | None = None
    memory_bytes: int | None = None
    cpu_time_seconds: typing.Any | None = None
    cpus: str | None = None
    read_only_rootfs: bool = False
    user: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            msg = "memory_bytes must be positive if provided."
            raise ValueError(msg)
        if self.cpu_time_seconds is not None:
            msg = (
                "DockerExecutionPolicy does not support cpu_time_seconds; configure CPU limits "
                "using Docker run options such as '--cpus'."
            )
            raise RuntimeError(msg)
        if self.cpus is not None and not self.cpus.strip():
            msg = "cpus must be a non-empty string when provided."
            raise ValueError(msg)
        if self.user is not None and not self.user.strip():
            msg = "user must be a non-empty string when provided."
            raise ValueError(msg)
        self.extra_run_args = tuple(self.extra_run_args or ())

    def spawn(
        self,
        *,
        workspace: Path,
        env: Mapping[str, str],
        command: Sequence[str],
    ) -> subprocess.Popen[str]:
        full_command = self._build_command(workspace, env, command)
        host_env = os.environ.copy()
        return _launch_subprocess(
            full_command,
            env=host_env,
            cwd=workspace,
            preexec_fn=None,
            start_new_session=False,
        )

    def _build_command(
        self,
        workspace: Path,
        env: Mapping[str, str],
        command: Sequence[str],
    ) -> list[str]:
        """Assemble the full `docker run` invocation.

        Args:
            workspace: Host directory to optionally bind-mount.
            env: Environment variables to forward into the container.
            command: The user command to execute inside the container.

        Returns:
            The complete argument list for `subprocess.Popen`.
        """
        binary = self._resolve_binary()
        full_command: list[str] = [binary, "run", "-i"]
        if self.remove_container_on_exit:
            full_command.append("--rm")
        if not self.network_enabled:
            full_command.extend(["--network", "none"])
        if self.memory_bytes is not None:
            full_command.extend(["--memory", str(self.memory_bytes)])
        if self._should_mount_workspace(workspace):
            host_path = str(workspace)
            full_command.extend(["-v", f"{host_path}:{host_path}"])
            full_command.extend(["-w", host_path])
        else:
            full_command.extend(["-w", "/"])
        if self.read_only_rootfs:
            full_command.append("--read-only")
        for key, value in env.items():
            full_command.extend(["-e", f"{key}={value}"])
        if self.cpus is not None:
            full_command.extend(["--cpus", self.cpus])
        if self.user is not None:
            full_command.extend(["--user", self.user])
        if self.extra_run_args:
            full_command.extend(self.extra_run_args)
        full_command.append(self.image)
        full_command.extend(command)
        return full_command

    @staticmethod
    def _should_mount_workspace(workspace: Path) -> bool:
        """Determine whether the workspace should be bind-mounted into the container.

        Ephemeral directories created by the shell middleware (prefixed with
        `langchain-shell-`) are not mounted to minimise host exposure.

        Args:
            workspace: The candidate workspace path.

        Returns:
            `True` if the workspace should be mounted.
        """
        return not workspace.name.startswith(SHELL_TEMP_PREFIX)

    def _resolve_binary(self) -> str:
        """Locate the Docker CLI binary on `PATH`.

        Returns:
            The absolute path to the resolved binary.

        Raises:
            RuntimeError: If the binary is not found.
        """
        path = shutil.which(self.binary)
        if path is None:
            msg = (
                "Docker execution policy requires the '%s' CLI to be installed"
                " and available on PATH."
            )
            raise RuntimeError(msg % self.binary)
        return path


__all__ = [
    "BaseExecutionPolicy",
    "CodexSandboxExecutionPolicy",
    "DockerExecutionPolicy",
    "HostExecutionPolicy",
]
