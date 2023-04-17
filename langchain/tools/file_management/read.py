from typing import Type

from pydantic import BaseModel, Field

from langchain.tools.base import BaseTool


class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="name of file")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    tool_args: Type[BaseModel] = WriteFileInput
    description: str = "Read file from disk"

    def _run(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, tool_input: str) -> str:
        # TODO: Add aiofiles method
        raise NotImplementedError
