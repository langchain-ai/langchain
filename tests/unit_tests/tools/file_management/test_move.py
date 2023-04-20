"""Test the FileMove tool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools.file_management.move import MoveFileTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
)


def test_move_file_with_root_dir() -> None:
    """Test the FileMove tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = MoveFileTool(root_dir=temp_dir)
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run({"source_path": "source.txt", "destination_path": "destination.txt"})
        assert not source_file.exists()
        assert destination_file.exists()
        assert destination_file.read_text() == "Hello, world!"


def test_move_file_errs_outside_root_dir() -> None:
    """Test the FileMove tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = MoveFileTool(root_dir=temp_dir)
        result = tool.run(
            {
                "source_path": "../source.txt",
                "destination_path": "../destination.txt",
            }
        )
        assert result == INVALID_PATH_TEMPLATE.format(
            arg_name="source_path", value="../source.txt"
        )


def test_move_file() -> None:
    """Test the FileMove tool."""
    with TemporaryDirectory() as temp_dir:
        tool = MoveFileTool()
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run(
            {"source_path": str(source_file), "destination_path": str(destination_file)}
        )
        assert not source_file.exists()
        assert destination_file.exists()
        assert destination_file.read_text() == "Hello, world!"
