import os
from typing import List

from langchain.tools.base import ArgInfo, BaseTool


class WriteFileTool(BaseTool):
    name: str = "write_file"
    tool_args: List[ArgInfo] = [
        ArgInfo(name="file_path", description="name of file"),
        ArgInfo(name="text", description="text to write to file"),
    ]
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

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError
