"""A tool for running python code in a REPL."""

import ast
from io import StringIO
import sys
from typing import Dict, Optional
import traceback

from pydantic import Field, root_validator

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
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`."
    )
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.python_repl.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("PythonReplTool does not support async")


class PythonAstREPLTool(BaseTool):
    """A tool for running python code in a REPL."""

    name = "python_repl_ast"
    description = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "When using this tool, sometimes output is abbreviated - "
        "make sure it does not look abbreviated before using it in your answer."
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)

    @root_validator(pre=True)
    def validate_python_version(cls, values: Dict) -> Dict:
        """Validate valid python version."""
        if sys.version_info < (3, 9):
            raise ValueError(
                "This tool relies on Python 3.9 or higher "
                "(as it uses new functionality in the `ast` module, "
                f"you have Python version: {sys.version}"
            )
        return values

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            '''
            Let's try to handle the following cases for the final statement:
                1. result # we ignore any stdout as a result of evaluating this expression
                2. function(input) # Returns `None`, but prints to stdout -> capture stdout
                3. throws exception: my_function(input) # raises `ValueError()`
                4. a statement that does not fit our desired syntax: e.g. `x = 1` or `x = print(5)`
            We can handle 1-2 but will throw an exception for 3-4.
            4 is handled the same way as 3.
            '''
            try:
                res = eval(module_end_str, self.globals, self.locals)
            finally:
                sys.stdout = old_stdout
            if res is not None and res != '':
                return str(res)
            # Try to capture stdout
            return mystdout.getvalue()
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("PythonReplTool does not support async")
