from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.tools.base import BaseTool
from langchain.tools.file_management.utils import get_validated_relative_path


class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="name of file")
    text: str = Field(..., description="text to write to file")


class WriteFileTool(BaseTool):
    name: str = "write_file"
    args_schema: Type[BaseModel] = WriteFileInput
    description: str = "Write file to disk"
    root_dir: Optional[str] = None
    """Directory to write file to.

    If specified, raises an error for file_paths oustide root_dir."""

    def _run(self, file_path: str, text: str) -> str:
        write_path = (
            get_validated_relative_path(Path(self.root_dir), file_path)
            if self.root_dir
            else Path(file_path)
        )
        try:
            write_path.parent.mkdir(exist_ok=True, parents=False)
            with write_path.open("w", encoding="utf-8") as f:
                f.write(text)
            return f"File written successfully to {file_path}."
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, file_path: str, text: str) -> str:
        # TODO: Add aiofiles method
        raise NotImplementedError
