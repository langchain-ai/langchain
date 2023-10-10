import warnings
from typing import List

import pytest

from langchain.tools.shell.tool import ShellInput, ShellTool

# Test data
test_commands = ["echo 'Hello, World!'", "echo 'Another command'"]


def test_shell_input_validation() -> None:
    shell_input = ShellInput(commands=test_commands)
    assert isinstance(shell_input.commands, list)
    assert len(shell_input.commands) == 2

    with warnings.catch_warnings(record=True) as w:
        ShellInput(commands=test_commands)
        assert len(w) == 1
        assert (
            str(w[-1].message)
            == "The shell tool has no safeguards by default. Use at your own risk."
        )


class PlaceholderProcess:
    def __init__(self, output: str = "") -> None:
        self._commands: List[str] = []
        self.output = output

    def _run(self, commands: List[str]) -> str:
        self._commands = commands
        return self.output

    def run(self, commands: List[str]) -> str:
        return self._run(commands)

    async def arun(self, commands: List[str]) -> str:
        return self._run(commands)


def test_shell_tool_init() -> None:
    placeholder = PlaceholderProcess()
    shell_tool = ShellTool(process=placeholder)
    assert shell_tool.name == "terminal"
    assert isinstance(shell_tool.description, str)
    assert shell_tool.args_schema == ShellInput
    assert shell_tool.process is not None


def test_shell_tool_run() -> None:
    placeholder = PlaceholderProcess(output="hello")
    shell_tool = ShellTool(process=placeholder)
    result = shell_tool._run(commands=test_commands)
    assert result.strip() == "hello"


@pytest.mark.asyncio
async def test_shell_tool_arun() -> None:
    placeholder = PlaceholderProcess(output="hello")
    shell_tool = ShellTool(process=placeholder)
    result = await shell_tool._arun(commands=test_commands)
    assert result.strip() == "hello"


def test_shell_tool_run_str() -> None:
    placeholder = PlaceholderProcess(output="hello")
    shell_tool = ShellTool(process=placeholder)
    result = shell_tool._run(commands="echo 'Hello, World!'")
    assert result.strip() == "hello"
