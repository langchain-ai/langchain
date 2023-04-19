import os
from typing import Type

from pydantic import BaseModel, Field

from langchain.tools.base import BaseTool


class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="name of file")
    text: str = Field(..., description="text to write to file")


class WriteFileTool(BaseTool):
    name: str = "write_file"
    tool_args: Type[BaseModel] = WriteFileInput
    description: str = "Write file to disk"

    def _run(self, file_path: str, text: str) -> str:
        try:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory) and directory:
                os.makedirs(directory)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            return "File written to successfully."
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, file_path: str, text: str) -> str:
        # TODO: Add aiofiles method
        raise NotImplementedError
