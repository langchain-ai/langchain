"""Wrapper around subprocess to run commands."""
from __future__ import annotations

import platform
import re
import subprocess
from typing import TYPE_CHECKING, List, Union
from uuid import uuid4

if TYPE_CHECKING:
    import pexpect


def _lazy_import_pexpect() -> pexpect:
    """Import pexpect only when needed."""
    if platform.system() == "Windows":
        raise ValueError("Persistent bash processes are not yet supported on Windows.")
    try:
        import pexpect

    except ImportError:
        raise ImportError(
            "pexpect required for persistent bash processes."
            " To install, run `pip install pexpect`."
        )
    return pexpect


class BashProcess:
    """Executes bash commands and returns the output."""

    def __init__(
        self,
        strip_newlines: bool = False,
        return_err_output: bool = False,
        persistent: bool = False,
    ):
        """Initialize with stripping newlines."""
        self.strip_newlines = strip_newlines
        self.return_err_output = return_err_output
        self.prompt = ""
        self.process = None
        if persistent:
            self.prompt = str(uuid4())
            self.process = self._initialize_persistent_process(self.prompt)

    @staticmethod
    def _initialize_persistent_process(prompt: str) -> pexpect.spawn:
        # Start bash in a clean environment
        # Doesn't work on windows
        pexpect = _lazy_import_pexpect()
        process = pexpect.spawn(
            "env", ["-i", "bash", "--norc", "--noprofile"], encoding="utf-8"
        )
        # Set the custom prompt
        process.sendline("PS1=" + prompt)

        process.expect_exact(prompt, timeout=10)
        return process

    def run(self, commands: Union[str, List[str]]) -> str:
        """Run commands and return final output."""
        if isinstance(commands, str):
            commands = [commands]
        commands = ";".join(commands)
        if self.process is not None:
            return self._run_persistent(
                commands,
            )
        else:
            return self._run(commands)

    def _run(self, command: str) -> str:
        """Run commands and return final output."""
        try:
            output = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout.decode()
        except subprocess.CalledProcessError as error:
            if self.return_err_output:
                return error.stdout.decode()
            return str(error)
        if self.strip_newlines:
            output = output.strip()
        return output

    def process_output(self, output: str, command: str) -> str:
        # Remove the command from the output using a regular expression
        pattern = re.escape(command) + r"\s*\n"
        output = re.sub(pattern, "", output, count=1)
        return output.strip()

    def _run_persistent(self, command: str) -> str:
        """Run commands and return final output."""
        pexpect = _lazy_import_pexpect()
        if self.process is None:
            raise ValueError("Process not initialized")
        self.process.sendline(command)

        # Clear the output with an empty string
        self.process.expect(self.prompt, timeout=10)
        self.process.sendline("")

        try:
            self.process.expect([self.prompt, pexpect.EOF], timeout=10)
        except pexpect.TIMEOUT:
            return f"Timeout error while executing command {command}"
        if self.process.after == pexpect.EOF:
            return f"Exited with error status: {self.process.exitstatus}"
        output = self.process.before
        output = self.process_output(output, command)
        if self.strip_newlines:
            return output.strip()
        return output
