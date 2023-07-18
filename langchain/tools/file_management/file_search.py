import fnmatch
import os
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)


class FileSearchInput(BaseModel):
    """Input for FileSearchTool."""

    dir_path: str = Field(
        default=".",
        description="Subdirectory to search in.",
    )
    pattern: str = Field(
        ...,
        description="Unix shell regex, where * matches everything.",
    )


class FileSearchTool(BaseFileToolMixin, BaseTool):
    """Tool that searches for files in a subdirectory that match a regex pattern."""

    name: str = "file_search"
    args_schema: Type[BaseModel] = FileSearchInput
    description: str = (
        "Recursively search for files in a subdirectory that match the regex pattern"
    )

    def _run(
        self,
        pattern: str,
        dir_path: str = ".",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            dir_path_ = self.get_relative_path(dir_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="dir_path", value=dir_path)
        matches = []
        try:
            for root, _, filenames in os.walk(dir_path_):
                for filename in fnmatch.filter(filenames, pattern):
                    absolute_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(absolute_path, dir_path_)
                    matches.append(relative_path)
            if matches:
                return "\n".join(matches)
            else:
                return f"No files found for pattern {pattern} in directory {dir_path}"
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(
        self,
        dir_path: str,
        pattern: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        # TODO: Add aiofiles method
        raise NotImplementedError
