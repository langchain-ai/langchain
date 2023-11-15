import shutil
from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)


class FileMoveInput(BaseModel):
    """Input for MoveFileTool."""

    source_path: str = Field(..., description="Path of the file to move")
    destination_path: str = Field(..., description="New path for the moved file")


class MoveFileTool(BaseFileToolMixin, BaseTool):
    """Tool that moves a file."""

    name: str = "move_file"
    args_schema: Type[BaseModel] = FileMoveInput
    description: str = "Move or rename a file from one location to another"

    def _run(
        self,
        source_path: str,
        destination_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            source_path_ = self.get_relative_path(source_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(
                arg_name="source_path", value=source_path
            )
        try:
            destination_path_ = self.get_relative_path(destination_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(
                arg_name="destination_path_", value=destination_path_
            )
        if not source_path_.exists():
            return f"Error: no such file or directory {source_path}"
        try:
            # shutil.move expects str args in 3.8
            shutil.move(str(source_path_), destination_path_)
            return f"File moved successfully from {source_path} to {destination_path}."
        except Exception as e:
            return "Error: " + str(e)

    # TODO: Add aiofiles method
