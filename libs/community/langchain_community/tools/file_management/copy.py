import shutil
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)


class FileCopyInput(BaseModel):
    """Input for CopyFileTool."""

    source_path: str = Field(..., description="Path of the file to copy")
    destination_path: str = Field(..., description="Path to save the copied file")


class CopyFileTool(BaseFileToolMixin, BaseTool):
    """Tool that copies a file."""

    name: str = "copy_file"
    args_schema: Type[BaseModel] = FileCopyInput
    description: str = "Create a copy of a file in a specified location"

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
                arg_name="destination_path", value=destination_path
            )
        try:
            shutil.copy2(source_path_, destination_path_, follow_symlinks=False)
            return f"File copied successfully from {source_path} to {destination_path}."
        except Exception as e:
            return "Error: " + str(e)

    # TODO: Add aiofiles method
