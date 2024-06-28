"""
Tool implementations for the Riza (https://riza.io) code interpreter API.

Documentation: https://docs.riza.io
API keys:      https://dashboard.riza.io
"""

from typing import Type, Optional, Any

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool, ToolException

from rizaio import Riza

class ExecPythonInput(BaseModel):
    code: str = Field(description="the Python code to execute")

class ExecPython(BaseTool):
    name: str = "riza_exec_python"
    description: str = """Execute Python code to solve problems.

    The Python runtime does not have filesystem access. You can use the httpx
    or requests library to make HTTP requests. Always print output to stdout."""
    args_schema: Type[BaseModel] = ExecPythonInput
    handle_tool_error: bool = True

    client: Riza = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Riza()

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        output = self.client.command.exec(language="PYTHON", code=code)
        if output.exit_code > 0:
            raise ToolException(f"Riza code execution returned this error with a non-zero exit code: {output.stderr}")
        return output.stdout

class ExecJavaScriptInput(BaseModel):
    code: str = Field(description="the JavaScript code to execute")

class ExecJavaScript(BaseTool):
    name: str = "riza_exec_javascript"
    description: str = """Execute JavaScript code to solve problems.

    The JavaScript runtime does not have filesystem access, but can use fetch
    to make HTTP requests and does include the global JSON object. Always print
    output to stdout."""
    args_schema: Type[BaseModel] = ExecJavaScriptInput
    handle_tool_error: bool = True

    client: Riza = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Riza()

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        output = self.client.command.exec(language="JAVASCRIPT", code=code)
        if output.exit_code > 0:
            raise ToolException(f"Riza code execution returned this error with a non-zero exit code: {output.stderr}")
        return output.stdout
