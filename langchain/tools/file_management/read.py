from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.tools.base import BaseTool
from langchain.tools.file_management.utils import get_validated_relative_path


class ReadFileInput(BaseModel):
    """Input for ReadFileTool."""

    file_path: str = Field(..., description="name of file")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    args_schema: Type[BaseModel] = ReadFileInput
    description: str = "Read file from disk"
    root_dir: Optional[str] = None
    """Directory to read file from.

    If specified, raises an error for file_paths oustide root_dir."""

    def _run(self, file_path: str) -> str:
        read_path = (
            get_validated_relative_path(Path(self.root_dir), file_path)
            if self.root_dir
            else Path(file_path)
        )
        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, tool_input: str) -> str:
        # TODO: Add aiofiles method
        raise NotImplementedError
