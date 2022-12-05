import subprocess
from typing import Dict, List

from langchain.chains.bash import BashChain

def test_pwd_command() -> None:
    """Test correct functionality."""
    chain = BashChain(input_key="command1", output_key="output1")
    command = "pwd"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": subprocess.check_output(command, shell=True).decode()}

    # Test with the more user-friendly interface.
    simple_output = chain.run(command)
    assert simple_output == subprocess.check_output(command, shell=True).decode()

def test_incorrect_command() -> None:
    """Test handling of incorrect command."""
    chain = BashChain(input_key="command1", output_key="output1")
    command = "invalid_command"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": "Command 'invalid_command' returned non-zero exit status 127."}

    # Test with the more user-friendly interface.
    simple_output = chain.run(command)
    assert simple_output == "Command 'invalid_command' returned non-zero exit status 127."


def test_create_dir_and_files() -> None:
    """Test creating a directory and adding files."""
    # Create a directory using BashChain.
    chain = BashChain(input_key="command1", output_key="output1")
    command = "mkdir test_dir"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": ""}

    # Check that the directory was created successfully.
    command = "ls"
    output = chain({"command1": command})
    assert "test_dir" in output["output1"]

    # Create two files in the directory using BashChain.
    command = "touch test_dir/test_file_1 test_dir/test_file_2"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": ""}

    # Check that the files were created successfully.
    command = "ls test_dir"
    output = chain({"command1": command})
    assert "test_file_1" in output["output1"]
    assert "test_file_2" in output["output1"]

    # Remove the directory and its contents using BashChain.
    command = "rm -r test_dir"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": ""}

    # Check that the directory and its contents were removed successfully.
    command = "ls"
    output = chain({"command1": command})
    assert "test_dir" not in output["output1"]

def test_ls_command() -> None:
    """Test correct functionality of ls command."""
    chain = BashChain(input_key="command1", output_key="output1")

    # Create a directory and two files.
    command = "mkdir test_directory && touch test_directory/file1.txt && touch test_directory/file2.txt"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": ""}

    # Check that the directory and files were created.
    command = "ls test_directory"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": "file1.txt\nfile2.txt\n"}

    # Delete the directory and files.
    command = "rm -rf test_directory"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": ""}

    # Check that the directory and files were deleted.
    command = "ls test_directory"
    output = chain({"command1": command})
    assert output == {"command1": command, "output1": "Command 'ls test_directory' returned non-zero exit status 1."}
