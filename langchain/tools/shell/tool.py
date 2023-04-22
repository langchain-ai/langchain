import warnings
from typing import List, Type

from pydantic import BaseModel, Field, root_validator


import platform

from langchain.tools.base import BaseTool
from langchain.utilities.bash import BashProcess


class ShellInput(BaseModel):
    """Commands for the Bash Shell tool."""

    commands: List[str] = Field(
        ...,
        description="List of shell commands to run. Deserialized using json.loads",
    )
    """List of shell commands to run."""
    restart_process: bool = Field(
        default=False,
        description="Whether to execute the commands in a new bash process.",
    )

    @root_validator
    def _validate_commands(cls, values: dict) -> dict:
        """Validate commands."""
        # TODO: Add real validators
        commands = values.get("commands")
        if not isinstance(commands, list):
            values["commands"] = [commands]
        # Warn that the bash tool is not safe
        warnings.warn(
            "The shell tool has no safeguards by default. Use at your own risk."
        )
        return values


def _get_default_bash_processs() -> BashProcess:
    """Get file path from string."""
    return BashProcess(return_err_output=True)


def _get_platform() -> str:
    """Get platform."""
    system = platform.system()
    if system == "Darwin":
        return "MacOS"
    return system


class ShellTool(BaseTool):
    """Tool to run shell commands."""

    process: BashProcess = Field(default_factory=_get_default_bash_processs)
    """Bash process to run commands."""

    name: str = "shell"
    """Name of tool."""

    description: str = f"Run shell commands on this {_get_platform()} machine."
    """Description of tool."""

    args_schema: Type[BaseModel] = ShellInput
    """Schema for input arguments."""

    def _restart_process(self) -> None:
        """Restart the bash process."""
        self.process = BashProcess(
            return_err_output=self.process.return_err_output,
            strip_newlines=self.process.strip_newlines,
        )

    def _run(self, commands: List[str], restart_process: bool = False) -> str:
        """Run commands and return final output."""
        if restart_process:
            self._restart_process()
        return self.process.run(commands)

    async def _arun(self, tool_input: str, restart_process: bool = False) -> str:
        """Run commands asynchronously and return final output."""
        if restart_process:
            self._restart_process()
        return await self.process.arun(tool_input)
