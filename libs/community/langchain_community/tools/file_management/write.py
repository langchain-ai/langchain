from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)


class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="name of file")
    text: str = Field(..., description="text to write to file")
    append: bool = Field(
        default=False, description="Whether to append to an existing file."
    )


class WriteFileTool(BaseFileToolMixin, BaseTool):  # type: ignore[override, override]
    """Tool that writes a file to disk."""

    name: str = "write_file"
    args_schema: Type[BaseModel] = WriteFileInput
    description: str = "Write file to disk"

    def _run(
        self,
        file_path: str,
        text: str,
        append: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            write_path = self.get_relative_path(file_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="file_path", value=file_path)
        try:
            write_path.parent.mkdir(exist_ok=True, parents=False)
            mode = "a" if append else "w"
            with write_path.open(mode, encoding="utf-8") as f:
                f.write(text)
            return f"File written successfully to {file_path}."
        except Exception as e:
            return "Error: " + str(e)

    # TODO: Add aiofiles method
