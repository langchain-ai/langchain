"""Test Python REPL Tools."""
import sys

import pytest

from langchain.tools.python.tool import PythonAstREPLTool, PythonREPLTool


def test_python_repl_tool_single_input() -> None:
    """Test that the python REPL tool works with a single input."""
    tool = PythonREPLTool()
    assert tool.is_single_input
    assert int(tool.run("print(1 + 1)").strip()) == 2


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_tool_single_input() -> None:
    """Test that the python REPL tool works with a single input."""
    tool = PythonAstREPLTool()
    assert tool.is_single_input
    assert tool.run("1 + 1") == 2
