"""A tool for running python code in a REPL."""

from pydantic import Field

from langchain.python import PythonREPL
from langchain.tools.base import BaseTool


def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)


class PythonREPLTool(BaseTool):
    """A tool for running python code in a REPL."""

    name = "Python REPL"
    description = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you expect output it should be printed out."
    )
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.python_repl.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("PythonReplTool does not support async")
