from typing import List

from langchain.tools.base import ArgInfo, BaseTool


class ReadFileTool(BaseTool):
    name: str = "read_file"
    tool_args: List[ArgInfo] = [ArgInfo(name="file_path", description="name of file")]
    description: str = "Read file from disk"

    def _run(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError
