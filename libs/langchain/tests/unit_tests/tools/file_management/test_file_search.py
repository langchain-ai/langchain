"""Test the FileSearch tool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools.file_management.file_search import FileSearchTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
)


def test_file_search_with_root_dir() -> None:
    """Test the FileSearch tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(root_dir=temp_dir)
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.log"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        matches = tool.run({"dir_path": ".", "pattern": "*.txt"}).split("\n")
        assert len(matches) == 1
        assert Path(matches[0]).name == "file1.txt"


def test_file_search_errs_outside_root_dir() -> None:
    """Test the FileSearch tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(root_dir=temp_dir)
        result = tool.run({"dir_path": "..", "pattern": "*.txt"})
        assert result == INVALID_PATH_TEMPLATE.format(arg_name="dir_path", value="..")


def test_file_search() -> None:
    """Test the FileSearch tool."""
    with TemporaryDirectory() as temp_dir:
        tool = FileSearchTool()
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.log"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        matches = tool.run({"dir_path": temp_dir, "pattern": "*.txt"}).split("\n")
        assert len(matches) == 1
        assert Path(matches[0]).name == "file1.txt"
