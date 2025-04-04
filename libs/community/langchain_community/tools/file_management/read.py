from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)


class ReadFileInput(BaseModel):
    """Input for ReadFileTool."""

    file_path: str = Field(..., description="name of file")


class ReadFileTool(BaseFileToolMixin, BaseTool):  # type: ignore[override, override]
    """Tool that reads a file."""

    name: str = "read_file"
    args_schema: Type[BaseModel] = ReadFileInput
    description: str = (
        "Read file from disk. This can read only files which are in text format, "
        "cannot interpret binary formats, pdfs, xlsx etc. "
        "The file path can be relative or absolute. "
        "If the file is not found, an error message is returned. "
        "If the file is a directory, an error message is returned."
    )

    def _run(
        self,
        file_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
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

    # TODO: Add aiofiles method
