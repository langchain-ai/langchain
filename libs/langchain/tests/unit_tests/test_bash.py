"""Test the bash utility."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from langchain.utilities.bash import BashProcess


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_pwd_command() -> None:
    """Test correct functionality."""
    session = BashProcess()
    commands = ["pwd"]
    output = session.run(commands)

    assert output == subprocess.check_output("pwd", shell=True).decode()


@pytest.mark.skip(reason="flaky on GHA, TODO to fix")
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
def test_incorrect_command() -> None:
    """Test handling of incorrect command."""
    session = BashProcess()
    output = session.run(["invalid_command"])
    assert output == "Command 'invalid_command' returned non-zero exit status 127."


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_incorrect_command_return_err_output() -> None:
    """Test optional returning of shell output on incorrect command."""
    session = BashProcess(return_err_output=True)
    output = session.run(["invalid_command"])
    assert re.match(
        r"^/bin/sh:.*invalid_command.*(?:not found|Permission denied).*$", output
    )


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_create_directory_and_files(tmp_path: Path) -> None:
    """Test creation of a directory and files in a temporary directory."""
    session = BashProcess(strip_newlines=True)

    # create a subdirectory in the temporary directory
    temp_dir = tmp_path / "test_dir"
    temp_dir.mkdir()

    # run the commands in the temporary directory
    commands = [
        f"touch {temp_dir}/file1.txt",
        f"touch {temp_dir}/file2.txt",
        f"echo 'hello world' > {temp_dir}/file2.txt",
        f"cat {temp_dir}/file2.txt",
    ]

    output = session.run(commands)
    assert output == "hello world"

    # check that the files were created in the temporary directory
    output = session.run([f"ls {temp_dir}"])
    assert output == "file1.txt\nfile2.txt"


@pytest.mark.skip(reason="flaky on GHA, TODO to fix")
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
