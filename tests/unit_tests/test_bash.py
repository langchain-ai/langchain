"""Test the bash utility."""
import subprocess
from pathlib import Path

from langchain.utilities.bash import BashProcess


def test_pwd_command() -> None:
    """Test correct functionality."""
    session = BashProcess()
    commands = ["pwd"]
    output = session.run(commands)

    assert output == subprocess.check_output("pwd", shell=True).decode()


def test_incorrect_command() -> None:
    """Test handling of incorrect command."""
    session = BashProcess()
    output = session.run(["invalid_command"])
    assert output == "Command 'invalid_command' returned non-zero exit status 127."


def test_incorrect_command_return_err_output() -> None:
    """Test optional returning of shell output on incorrect command."""
    session = BashProcess(return_err_output=True)
    output = session.run(["invalid_command"])
    assert output == "/bin/sh: 1: invalid_command: not found\n"


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
