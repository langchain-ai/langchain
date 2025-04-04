"""
Tool implementations for the Riza (https://riza.io) code interpreter API.

Documentation: https://docs.riza.io
API keys:      https://dashboard.riza.io
"""

from typing import Any, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field


class ExecPythonInput(BaseModel):
    code: str = Field(description="the Python code to execute")


class ExecPython(BaseTool):  # type: ignore[override, override]
    """Riza Code tool.

    Setup:
        Install ``langchain-community`` and ``rizaio`` and set environment variable ``RIZA_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community rizaio
            export RIZA_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.riza.command import ExecPython

            tool = ExecPython()

    Invocation with args:
        .. code-block:: python

            tool.invoke("x = 5; print(x)")

        .. code-block:: python

            '5\\n'

    Invocation with ToolCall:

        .. code-block:: python

            tool.invoke({"args": {"code":"x = 5; print(x)"}, "id": "1", "name": tool.name, "type": "tool_call"})

        .. code-block:: python

            tool.invoke({"args": {"code":"x = 5; print(x)"}, "id": "1", "name": tool.name, "type": "tool_call"})

    """  # noqa: E501

    name: str = "riza_exec_python"
    description: str = """Execute Python code to solve problems.

    The Python runtime does not have filesystem access. You can use the httpx
    or requests library to make HTTP requests. Always print output to stdout."""
    args_schema: Type[BaseModel] = ExecPythonInput
    handle_tool_error: bool = True

    client: Any = None

    def __init__(self, **kwargs: Any) -> None:
        try:
            from rizaio import Riza
        except ImportError as e:
            raise ImportError(
                "Couldn't import the `rizaio` package. "
                "Try running `pip install rizaio`."
            ) from e
        super().__init__(**kwargs)
        self.client = Riza()

    def _run(
        self, code: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        output = self.client.command.exec(language="PYTHON", code=code)
        if output.exit_code > 0:
            raise ToolException(
                f"Riza code execution returned a non-zero exit code. "
                f"The output captured from stderr was:\n{output.stderr}"
            )
        return output.stdout


class ExecJavaScriptInput(BaseModel):
    code: str = Field(description="the JavaScript code to execute")


class ExecJavaScript(BaseTool):  # type: ignore[override, override]
    """A tool implementation to execute JavaScript via Riza's Code Interpreter API."""

    name: str = "riza_exec_javascript"
    description: str = """Execute JavaScript code to solve problems.

    The JavaScript runtime does not have filesystem access, but can use fetch
    to make HTTP requests and does include the global JSON object. Always print
    output to stdout."""
    args_schema: Type[BaseModel] = ExecJavaScriptInput
    handle_tool_error: bool = True

    client: Any = None

    def __init__(self, **kwargs: Any) -> None:
        try:
            from rizaio import Riza
        except ImportError as e:
            raise ImportError(
                "Couldn't import the `rizaio` package. "
                "Try running `pip install rizaio`."
            ) from e
        super().__init__(**kwargs)
        self.client = Riza()

    def _run(
        self, code: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        output = self.client.command.exec(language="JAVASCRIPT", code=code)
        if output.exit_code > 0:
            raise ToolException(
                f"Riza code execution returned a non-zero exit code. "
                f"The output captured from stderr was:\n{output.stderr}"
            )
        return output.stdout
