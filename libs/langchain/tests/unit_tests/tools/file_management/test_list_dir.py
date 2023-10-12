"""Test the DirectoryListing tool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools.file_management.list_dir import ListDirectoryTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
)


def test_list_directory_with_root_dir() -> None:
    """Test the DirectoryListing tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = ListDirectoryTool(root_dir=temp_dir)
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.txt"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        entries = tool.run({"dir_path": "."}).split("\n")
        assert set(entries) == {"file1.txt", "file2.txt"}


def test_list_directory_errs_outside_root_dir() -> None:
    """Test the DirectoryListing tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = ListDirectoryTool(root_dir=temp_dir)
        result = tool.run({"dir_path": ".."})
        assert result == INVALID_PATH_TEMPLATE.format(arg_name="dir_path", value="..")


def test_list_directory() -> None:
    """Test the DirectoryListing tool."""
    with TemporaryDirectory() as temp_dir:
        tool = ListDirectoryTool()
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.txt"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        entries = tool.run({"dir_path": temp_dir}).split("\n")
        assert set(entries) == {"file1.txt", "file2.txt"}
