from typing import Type

from pydantic import BaseModel, Field

from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileTool,
    FileValidationError,
)


class EditFileInput(BaseModel):
    """Input for editing a file by line"""

    file_path: str = Field(..., description="name of file")
    start_line: int = Field(..., description="line number of starting line")
    end_line: int = Field(..., description="line number of ending line")
    text: str = Field(..., description="text to insert. (separate lines with \n)")


class EditFileTool(BaseFileTool):
    name: str = "edit_file"
    args_schema: Type[BaseModel] = EditFileInput
    description: str = "Edit file between line numbers (line numbers start at 0)."

    def _run(self, file_path: str, start_line: int, end_line: int, text: str) -> str:
        try:
            read_path = self.get_relative_path(file_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="file_path", value=file_path)

        if not read_path.exists():
            return f"Error: no such file or directory: {file_path}"

        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.readlines()

            content[start_line:end_line] = text.splitlines()

            with read_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(content))
            return f"{file_path} changed successfully!"

        except Exception as e:
            print(e)
            return "Error: " + str(e)

    async def _arun(
        self, file_path: str, start_line: int, end_line: int, text: str
    ) -> str:
        raise NotImplementedError
