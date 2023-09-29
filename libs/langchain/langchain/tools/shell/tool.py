import asyncio
import platform
import warnings
from typing import List, Optional, Type, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.tools.base import BaseTool
from langchain.utilities.bash import BashProcess


class ShellInput(BaseModel):
    """Commands for the Bash Shell tool."""

    commands: Union[str, List[str]] = Field(
        ...,
        description="List of shell commands to run. Deserialized using json.loads",
    )
    """List of shell commands to run."""

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

    name: str = "terminal"
    """Name of tool."""

    description: str = f"Run shell commands on this {_get_platform()} machine."
    """Description of tool."""

    args_schema: Type[BaseModel] = ShellInput
    """Schema for input arguments."""

    def _run(
        self,
        commands: Union[str, List[str]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run commands and return final output."""
        return self.process.run(commands)

    async def _arun(
        self,
        commands: Union[str, List[str]],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run commands asynchronously and return final output."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process.run, commands
        )
