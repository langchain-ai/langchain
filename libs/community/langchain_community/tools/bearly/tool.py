import base64
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Type

import requests
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools import Tool


def strip_markdown_code(md_string: str) -> str:
    """Strip markdown code from a string."""
    stripped_string = re.sub(r"^`{1,3}.*?\n", "", md_string, flags=re.DOTALL)
    stripped_string = re.sub(r"`{1,3}$", "", stripped_string)
    return stripped_string


def head_file(path: str, n: int) -> List[str]:
    """Get the first n lines of a file."""
    try:
        with open(path, "r") as f:
            return [str(line) for line in itertools.islice(f, n)]
    except Exception:
        return []


def file_to_base64(path: str) -> str:
    """Convert a file to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class BearlyInterpreterToolArguments(BaseModel):
    """Arguments for the BearlyInterpreterTool."""

    python_code: str = Field(
        ...,
        example="print('Hello World')",
        description=(
            "The pure python script to be evaluated. "
            "The contents will be in main.py. "
            "It should not be in markdown format."
        ),
    )


base_description = """Evaluates python code in a sandbox environment. \
The environment resets on every execution. \
You must send the whole script every time and print your outputs. \
Script should be pure python code that can be evaluated. \
It should be in python format NOT markdown. \
The code should NOT be wrapped in backticks. \
All python packages including requests, matplotlib, scipy, numpy, pandas, \
etc are available. \
If you have any files outputted write them to "output/" relative to the execution \
path. Output can only be read from the directory, stdout, and stdin. \
Do not use things like plot.show() as it will \
not work instead write them out `output/` and a link to the file will be returned. \
print() any output and results so you can capture the output."""  # noqa: T201


class FileInfo(BaseModel):
    """Information about a file to be uploaded."""

    source_path: str
    description: str
    target_path: str


class BearlyInterpreterTool:
    """Tool for evaluating python code in a sandbox environment."""

    api_key: str
    endpoint = "https://exec.bearly.ai/v1/interpreter"
    name = "bearly_interpreter"
    args_schema: Type[BaseModel] = BearlyInterpreterToolArguments
    files: Dict[str, FileInfo] = {}

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def file_description(self) -> str:
        if len(self.files) == 0:
            return ""
        lines = ["The following files available in the evaluation environment:"]
        for target_path, file_info in self.files.items():
            peek_content = head_file(file_info.source_path, 4)
            lines.append(
                f"- path: `{target_path}` \n first four lines: {peek_content}"
                f" \n description: `{file_info.description}`"
            )
        return "\n".join(lines)

    @property
    def description(self) -> str:
        return (base_description + "\n\n" + self.file_description).strip()

    def make_input_files(self) -> List[dict]:
        files = []
        for target_path, file_info in self.files.items():
            files.append(
                {
                    "pathname": target_path,
                    "contentsBasesixtyfour": file_to_base64(file_info.source_path),
                }
            )
        return files

    def _run(self, python_code: str) -> dict:
        script = strip_markdown_code(python_code)
        resp = requests.post(
            "https://exec.bearly.ai/v1/interpreter",
            data=json.dumps(
                {
                    "fileContents": script,
                    "inputFiles": self.make_input_files(),
                    "outputDir": "output/",
                    "outputAsLinks": True,
                }
            ),
            headers={"Authorization": self.api_key},
        ).json()
        return {
            "stdout": (
                base64.b64decode(resp["stdoutBasesixtyfour"]).decode()
                if resp["stdoutBasesixtyfour"]
                else ""
            ),
            "stderr": (
                base64.b64decode(resp["stderrBasesixtyfour"]).decode()
                if resp["stderrBasesixtyfour"]
                else ""
            ),
            "fileLinks": resp["fileLinks"],
            "exitCode": resp["exitCode"],
        }

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

    def add_file(self, source_path: str, target_path: str, description: str) -> None:
        if target_path in self.files:
            raise ValueError("target_path already exists")
        if not Path(source_path).exists():
            raise ValueError("source_path does not exist")
        self.files[target_path] = FileInfo(
            target_path=target_path, source_path=source_path, description=description
        )

    def clear_files(self) -> None:
        self.files = {}

    # TODO: this is because we can't have a dynamic description
    #  because of the base pydantic class
    def as_tool(self) -> Tool:
        return Tool.from_function(
            func=self._run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )
