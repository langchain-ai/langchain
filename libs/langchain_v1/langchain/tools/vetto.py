"""Vetto sandboxed process and shell execution tools."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import run_in_executor
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )

logger = logging.getLogger(__name__)


class VettoProcessInput(BaseModel):
    """Input schema for Vetto sandboxed execution."""

    command: str = Field(
        ...,
        description="Shell command string to execute safely inside the sandbox.",
    )
    cwd: str | None = Field(
        default=None,
        description="Optional working directory inside the sandbox workspace.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables for the sandboxed process.",
    )
    timeout: int | None = Field(
        default=None,
        description="Optional execution timeout in seconds for this invocation.",
    )


class VettoProcessTool(BaseTool):
    """Execute shell commands safely within an unprivileged kernel-level Vetto sandbox.

    Vetto provides daemon-less, rootless sandbox boundaries (Linux Landlock LSM
    ABI 1-6, namespaces, cgroups v2, macOS Seatbelt, Windows LPAC) with sub-4ms
    cold startup overhead. It masks sensitive credentials (~/.ssh, ~/.aws, .env)
    at the VFS layer and enforces strict network and resource limits.
    """

    name: str = "vetto_process"
    description: str = (
        "Execute a shell command inside an unprivileged kernel-level Vetto sandbox with "
        "Landlock LSM / macOS Seatbelt isolation and zero container overhead. "
        "Enforces credential masking, process-tree cleanup, and fail-closed timeout handling."
    )
    args_schema: type[BaseModel] = VettoProcessInput

    working_dir: str | None = Field(
        default=None,
        description="Root workspace directory for filesystem sandbox containment.",
    )
    net: str = Field(
        default="off",
        description="Network isolation mode: 'off' (default, airgapped), 'allowlist', or 'host'.",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="List of permitted domain names when net='allowlist'.",
    )
    allow_write: list[str] | None = Field(
        default=None,
        description="Additional file or directory paths permitted for write access.",
    )
    allow_read: list[str] | None = Field(
        default=None,
        description="Additional file or directory paths permitted for read-only access.",
    )
    timeout: int | None = Field(
        default=120,
        description="Default execution timeout in seconds.",
    )
    memory_limit: str | None = Field(
        default=None,
        description="Optional memory limit ceiling (e.g. '512MB', '1GB').",
    )
    vetto_binary: str | None = Field(
        default=None,
        description="Explicit path to the vetto binary. Defaults to autodetecting in PATH.",
    )
    allow_fallback: bool = Field(
        default=False,
        description=(
            "If True, fall back to process-group isolated execution when vetto is not found."
        ),
    )

    def _resolve_vetto_binary(self) -> str | None:
        """Resolve path to the vetto executable or None if not installed."""
        if (
            self.vetto_binary
            and Path(self.vetto_binary).is_file()
            and os.access(self.vetto_binary, os.X_OK)
        ):
            return self.vetto_binary

        env_path = os.getenv("VETTO_PATH")
        if env_path and Path(env_path).is_file() and os.access(env_path, os.X_OK):
            return env_path

        found = shutil.which("vetto")
        if found:
            return found

        candidates = [
            str(Path("~/.cargo/bin/vetto").expanduser()),
            "/usr/local/bin/vetto",
            "/usr/bin/vetto",
        ]
        for path in candidates:
            if Path(path).is_file() and os.access(path, os.X_OK):
                return path

        return None

    def _build_command(
        self,
        command_args: list[str],
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> list[str]:
        """Construct the sandboxed command vector prefixed with vetto CLI parameters."""
        vetto_bin = self._resolve_vetto_binary()
        if not vetto_bin:
            if not self.allow_fallback:
                msg = (
                    "Vetto binary not found. Install via 'cargo install vetto' "
                    "or 'npm install -g @shledery/vetto', set VETTO_PATH, or "
                    "configure allow_fallback=True."
                )
                raise RuntimeError(msg)
            return command_args

        effective_cwd = cwd or self.working_dir or str(Path.cwd())
        resolved_cwd = str(Path(effective_cwd).resolve())

        run_args = [vetto_bin, "run", f"--net={self.net}"]

        effective_timeout = timeout if timeout is not None else self.timeout
        if effective_timeout:
            run_args.extend(["--timeout", str(effective_timeout)])

        if self.memory_limit:
            run_args.extend(["--memory", self.memory_limit])

        run_args.extend(["--allow-write", resolved_cwd])

        if self.allow_write:
            for p in self.allow_write:
                run_args.extend(["--allow-write", str(Path(p).resolve())])

        if self.allow_read:
            for p in self.allow_read:
                run_args.extend(["--allow-read", str(Path(p).resolve())])

        if self.net == "allowlist" and self.allowed_domains:
            for domain in self.allowed_domains:
                run_args.extend(["--allow-domain", domain])

        run_args.append("--")
        run_args.extend(command_args)
        return run_args

    def _execute_subprocess(
        self,
        command_args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute command within sandbox process boundary with group cleanup."""
        full_command = self._build_command(command_args, cwd=cwd, timeout=timeout)
        effective_cwd = cwd or self.working_dir or str(Path.cwd())
        resolved_cwd = str(Path(effective_cwd).resolve())

        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        effective_timeout = timeout if timeout is not None else self.timeout

        kwargs: dict[str, Any] = {
            "cwd": resolved_cwd,
            "env": exec_env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }

        if hasattr(os, "setsid"):
            kwargs["preexec_fn"] = os.setsid

        start_time = time.monotonic()
        try:
            proc = subprocess.Popen(full_command, **kwargs)  # noqa: S603
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=effective_timeout)
                elapsed = time.monotonic() - start_time
                return {
                    "exit_code": proc.returncode,
                    "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                    "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                    "timed_out": False,
                    "elapsed_seconds": round(elapsed, 4),
                }
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        proc.kill()
                else:
                    proc.kill()

                stdout_bytes, stderr_bytes = proc.communicate()
                out = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                err = (
                    stderr_bytes.decode("utf-8", errors="replace")
                    if stderr_bytes
                    else "Command timed out"
                )
                return {
                    "exit_code": 124,
                    "stdout": out,
                    "stderr": err,
                    "timed_out": True,
                    "elapsed_seconds": round(time.monotonic() - start_time, 4),
                }
        except Exception as exc:
            logger.exception("Failed to execute sandboxed command.")
            return {
                "exit_code": 125,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "elapsed_seconds": 0.0,
            }

    def _run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
    ) -> str:
        """Run shell command inside the Vetto sandbox and return output."""
        sandbox_root = Path(self.working_dir or Path.cwd()).resolve()
        if cwd:
            raw_path = Path(cwd)
            if raw_path.is_absolute():
                resolved_cwd = raw_path.resolve()
            else:
                resolved_cwd = (sandbox_root / raw_path).resolve()
            try:
                resolved_cwd.relative_to(sandbox_root)
            except ValueError:
                msg = f"Execution cwd '{cwd}' escapes configured workspace '{sandbox_root}'"
                raise PermissionError(msg) from None
            effective_cwd = str(resolved_cwd)
        else:
            effective_cwd = str(sandbox_root)

        if platform.system() == "Windows":
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["sh", "-c", command]

        result = self._execute_subprocess(
            shell_cmd,
            cwd=effective_cwd,
            env=env,
            timeout=timeout,
        )

        if result["timed_out"]:
            return f"Error: Command timed out after {timeout or self.timeout} seconds."

        if result["exit_code"] != 0:
            err = result["stderr"].strip()
            out = result["stdout"].strip()
            combined = f"{out}\n{err}".strip() if out and err else (out or err)
            return f"{combined}\n\nExit code: {result['exit_code']}".strip()

        return result["stdout"]

    async def _arun(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Asynchronously run shell command inside Vetto sandbox."""
        return await run_in_executor(
            None,
            self._run,
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            run_manager=run_manager,
        )


class VettoShellTool(VettoProcessTool):
    """Alternative alias for VettoProcessTool matching ShellTool nomenclature."""

    name: str = "vetto_shell"
    description: str = (
        "Execute a shell command inside a kernel-level Vetto sandbox with "
        "Landlock LSM / macOS Seatbelt isolation and zero container overhead."
    )
