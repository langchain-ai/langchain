"""Test the ReadFile tool."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools.file_management.edit import EditFileTool


def test_edit_file_with_root_dir() -> None:
    """Test the EditFileTool"""
    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "file.txt"
        with file_path.open("w") as f:
            f.write(
                """Hello, world!
            This world is good. I like somethings.
            You might also like somethings.
            If you don't you are allowed to make new things.
            Maybe your thing will not work out, but don't worry try something else.
            Need to write more to test this.
            """
            )

        tool = EditFileTool(root_dir=temp_dir)
        result = tool.run(
            {
                "file_path": str(file_path),
                "start_line": 1,
                "end_line": 2,
                "text": "Test1\nTest2",
            }
        )
        assert result == f"{str(file_path)} changed successfully!"
