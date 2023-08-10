"""Test the WriteFile tool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
)
from langchain.tools.file_management.write import WriteFileTool


def test_write_file_with_root_dir() -> None:
    """Test the WriteFile tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = WriteFileTool(root_dir=temp_dir)
        tool.run({"file_path": "file.txt", "text": "Hello, world!"})
        assert (Path(temp_dir) / "file.txt").exists()
        assert (Path(temp_dir) / "file.txt").read_text() == "Hello, world!"


def test_write_file_errs_outside_root_dir() -> None:
    """Test the WriteFile tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = WriteFileTool(root_dir=temp_dir)
        result = tool.run({"file_path": "../file.txt", "text": "Hello, world!"})
        assert result == INVALID_PATH_TEMPLATE.format(
            arg_name="file_path", value="../file.txt"
        )


def test_write_file() -> None:
    """Test the WriteFile tool."""
    with TemporaryDirectory() as temp_dir:
        file_path = str(Path(temp_dir) / "file.txt")
        tool = WriteFileTool()
        tool.run({"file_path": file_path, "text": "Hello, world!"})
        assert (Path(temp_dir) / "file.txt").exists()
        assert (Path(temp_dir) / "file.txt").read_text() == "Hello, world!"
