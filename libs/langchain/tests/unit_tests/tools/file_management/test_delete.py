"""Test the delete tool."""
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.tools import DeleteFileTool


def test_delete() -> None:
    delete_file_tool = DeleteFileTool()
    with TemporaryDirectory() as temp_dir:
        file_1 = Path(temp_dir) / "file1.txt"
        file_1.write_text("File 1 content")
        result = delete_file_tool._run(file_path=str(file_1))
        assert result == f"File deleted successfully: {file_1}."

        file_2 = Path(temp_dir) / "file2.txt"
        result = delete_file_tool._run(file_path=str(file_2))
        assert result == f"Error: no such file or directory: {file_2}"
