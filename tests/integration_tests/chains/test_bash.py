"""Test the bash utility."""
import subprocess
import sys
from pathlib import Path

import pytest

from langchain.utilities.bash import BashProcess


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_pwd_command_persistent() -> None:
    """Test correct functionality when the bash process is persistent."""
    session = BashProcess(persistent=True, strip_newlines=True)
    commands = ["pwd"]
    output = session.run(commands)

    assert subprocess.check_output("pwd", shell=True).decode().strip() in output

    session.run(["cd .."])
    new_output = session.run(["pwd"])
    # Assert that the new_output is a parent of the old output
    assert Path(output).parent == Path(new_output)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_create_bash_persistent() -> None:
    """Test the pexpect persistent bash terminal"""
    session = BashProcess(persistent=True)
    response = session.run("echo hello")
    response += session.run("echo world")

    assert "hello" in response
    assert "world" in response
