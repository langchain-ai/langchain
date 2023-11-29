import os
from typing import Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)


class FileDeleteInput(BaseModel):
    """Input for DeleteFileTool."""

    file_path: str = Field(..., description="Path of the file to delete")


class DeleteFileTool(BaseFileToolMixin, BaseTool):
    """Tool that deletes a file."""

    name: str = "file_delete"
    args_schema: Type[BaseModel] = FileDeleteInput
    description: str = "Delete a file"

    def _run(
        self,
        file_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            file_path_ = self.get_relative_path(file_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="file_path", value=file_path)
        if not file_path_.exists():
            return f"Error: no such file or directory: {file_path}"
        try:
            os.remove(file_path_)
            return f"File deleted successfully: {file_path}."
        except Exception as e:
            return "Error: " + str(e)

    # TODO: Add aiofiles method
