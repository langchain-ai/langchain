"""Wrapper around subprocess to run commands."""
import subprocess
from typing import List, Union


class WSLProcess:
    """Executes bash commands on Windows Subsystem for Linux (WSL) and returns the output."""

    def __init__(self, strip_newlines: bool = False):
        """Initialize with stripping newlines."""
        self.strip_newlines = strip_newlines

    def run(self, commands: Union[str, List[str]]) -> str:
        """Run commands and return final output."""
        if isinstance(commands, str):
            commands = [commands]
        commands = ";".join(commands)
        commands = "wsl.exe " + commands
        try:
            output = subprocess.check_output(commands, shell=True).decode()

        except subprocess.CalledProcessError as error:
            return str(error)
        if self.strip_newlines:
            output = output.strip()
        return output
