from typing import Type

from pydantic import BaseModel, Field

from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileTool,
    FileValidationError,
)


class ReadFileInput(BaseModel):
    """Input for ReadFileTool."""

    file_path: str = Field(..., description="name of file")


class ReadFileTool(BaseFileTool):
    name: str = "read_file"
    args_schema: Type[BaseModel] = ReadFileInput
    description: str = "Read file from disk"

    def _run(self, file_path: str) -> str:
        try:
            read_path = self.get_relative_path(file_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="file_path", value=file_path)
        if not read_path.exists():
            return f"Error: no such file or directory: {file_path}"
        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, file_path: str) -> str:
        # TODO: Add aiofiles method
        raise NotImplementedError
