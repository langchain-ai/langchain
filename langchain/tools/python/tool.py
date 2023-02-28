"""A tool for running python code in a REPL."""

from langchain.python import PythonREPL
from langchain.tools.base import BaseTool


class PythonREPLTool(BaseTool):
    """A tool for running python code in a REPL."""

    name = "Python REPL"
    description = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you expect output it should be printed out."
    )
    python_repl: PythonREPL = PythonREPL()

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.python_repl.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("PythonReplTool does not support async")
