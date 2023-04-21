"""Test the ReadFile tool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools.file_management.read import ReadFileTool


def test_read_file_with_root_dir() -> None:
    """Test the ReadFile tool."""
    with TemporaryDirectory() as temp_dir:
        with (Path(temp_dir) / "file.txt").open("w") as f:
            f.write("Hello, world!")
        tool = ReadFileTool(root_dir=temp_dir)
        result = tool.run("file.txt")
        assert result == "Hello, world!"
        # Check absolute files can still be passed if they lie within the root dir.
        result = tool.run(str(Path(temp_dir) / "file.txt"))
        assert result == "Hello, world!"


def test_read_file() -> None:
    """Test the ReadFile tool."""
    with TemporaryDirectory() as temp_dir:
        with (Path(temp_dir) / "file.txt").open("w") as f:
            f.write("Hello, world!")
        tool = ReadFileTool()
        result = tool.run(str(Path(temp_dir) / "file.txt"))
        assert result == "Hello, world!"
