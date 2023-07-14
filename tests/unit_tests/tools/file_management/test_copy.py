"""Test the FileCopy tool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools.file_management.copy import CopyFileTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
)


def test_copy_file_with_root_dir() -> None:
    """Test the FileCopy tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = CopyFileTool(root_dir=temp_dir)
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run({"source_path": "source.txt", "destination_path": "destination.txt"})
        assert source_file.exists()
        assert destination_file.exists()
        assert source_file.read_text() == "Hello, world!"
        assert destination_file.read_text() == "Hello, world!"


def test_copy_file_errs_outside_root_dir() -> None:
    """Test the FileCopy tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = CopyFileTool(root_dir=temp_dir)
        result = tool.run(
            {
                "source_path": "../source.txt",
                "destination_path": "../destination.txt",
            }
        )
        assert result == INVALID_PATH_TEMPLATE.format(
            arg_name="source_path", value="../source.txt"
        )


def test_copy_file() -> None:
    """Test the FileCopy tool."""
    with TemporaryDirectory() as temp_dir:
        tool = CopyFileTool()
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run(
            {"source_path": str(source_file), "destination_path": str(destination_file)}
        )
        assert source_file.exists()
        assert destination_file.exists()
        assert source_file.read_text() == "Hello, world!"
        assert destination_file.read_text() == "Hello, world!"
