import subprocess
from typing import Dict, List

from langchain.chains.bash import BashChain

def test_pwd_command() -> None:
    """Test correct functionality."""
    chain = BashChain(input_key="commands", output_key="outputs")
    commands = ["pwd"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": [subprocess.check_output("pwd", shell=True).decode()]}

    # Test with the more user-friendly interface.
    simple_output = chain.run([ "pwd" ])
    assert simple_output == [subprocess.check_output("pwd", shell=True).decode()]


def test_incorrect_command() -> None:
    """Test handling of incorrect command."""
    chain = BashChain(input_key="commands", output_key="outputs")
    commands = ["invalid_command"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": ["Command 'invalid_command' returned non-zero exit status 127."]}

    # Test with the more user-friendly interface.
    simple_output = chain.run([ "invalid_command" ])
    assert simple_output == ["Command 'invalid_command' returned non-zero exit status 127."]


def test_create_dir_and_files() -> None:
    """Test creating a directory and adding files."""
    # Create a directory using BashChain.
    chain = BashChain(input_key="commands", output_key="outputs")

    # Check if the test_dir directory already exists.
    commands = ["ls"]
    output = chain({"commands": commands})
    if "test_dir" in output["outputs"][0]:
        # If the test_dir directory already exists, delete it.
        commands = ["rm -r test_dir"]
        output = chain({"commands": commands})

    # Create the test_dir directory.
    commands = ["mkdir test_dir"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": [""]}


    # Check that the directory was created successfully.
    commands = ["ls"]
    output = chain({"commands": commands})
    assert "test_dir" in output["outputs"][0]

    # Create two files in the directory using BashChain.
    commands = ["touch test_dir/test_file_1", "touch test_dir/test_file_2", "ls test_dir"]
    output = chain({"commands": commands})
    assert output["outputs"][:2] == ["", ""]
    assert "test_file_1" in output["outputs"][-1]
    assert "test_file_2" in output["outputs"][-1]

    # Remove the directory and its contents using BashChain.
    commands = ["rm -r test_dir"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": [""]}

    # Check that the directory and its contents were removed successfully.
    commands = ["ls"]
    output = chain({"commands": commands})
    assert "test_dir" not in output["outputs"][0]


def test_ls_command() -> None:
    """Test correct functionality of ls command."""
    chain = BashChain(input_key="commands", output_key="outputs")

    # Create a directory and two files.
    commands = ["mkdir test_directory && touch test_directory/file1.txt && touch test_directory/file2.txt"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": [""]}

    # Check that the directory and files were created.
    commands = ["ls test_directory"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": ["file1.txt\nfile2.txt\n"]}

    # Delete the directory and files.
    commands = ["rm -rf test_directory"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": [""]}

    # Check that the directory and files were deleted.
    commands = ["ls test_directory"]
    output = chain({"commands": commands})
    assert output == {"commands": commands, "outputs": ["Command 'ls test_directory' returned non-zero exit status 1."]}
