from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.tools.file_management.utils import get_validated_relative_path
from langchain.tools.structured import BaseStructuredTool


class ReadFileInput(BaseModel):
    """Input for ReadFileTool."""

    file_path: Path = Field(..., description="name of file")


class ReadFileTool(BaseStructuredTool[ReadFileInput, str]):
    name: str = "read_file"
    args_schema: Type[ReadFileInput] = ReadFileInput
    description: str = "Read file from disk"
    root_dir: Optional[str] = None
    """Directory to read file from.

    If specified, raises an error for file_paths oustide root_dir."""

    def _run(self, tool_input: ReadFileInput) -> str:
        read_path = (
            get_validated_relative_path(Path(self.root_dir), tool_input.file_path)
            if self.root_dir
            else tool_input.file_path
        )
        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, tool_input: ReadFileInput) -> str:
        # TODO: Add aiofiles method
        raise NotImplementedError
