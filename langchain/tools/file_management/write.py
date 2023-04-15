import json
import os
from typing import Dict

from pydantic import root_validator

from langchain.tools.base import BaseMultiArgTool


def write_to_file(filename, text):
    """Write text to a file"""
    try:
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        return "File written to successfully."
    except Exception as e:
        return "Error: " + str(e)


class WriteFileTool(BaseMultiArgTool):
    name: str = "write_file"
    tool_args: Dict[str, str] = {"file": "<file>", "text": "<text>"}
    description: str = "Write file to disk"

    def _run(self, file, text) -> str:
        write_to_file(file, text)
        return "Done"

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError
