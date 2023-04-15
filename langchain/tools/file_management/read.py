from typing import Dict

from langchain.tools.base import BaseMultiArgTool


class ReadFileTool(BaseMultiArgTool):
    name: str = "read_file"
    tool_args: Dict[str, str] = {"file": "<file>"}
    description: str = "Read file from disk"

    def _run(self, file: str) -> str:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError
