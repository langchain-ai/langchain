"""Test functionality of Python REPL."""
import sys

import pytest

from langchain.tools.python.tool import PythonAstREPLTool, PythonREPLTool
from langchain.utilities import PythonREPL

_SAMPLE_CODE = """
```
def multiply():
    print(5*6)
multiply()
```
"""

_AST_SAMPLE_CODE = """
```
def multiply():
    return(5*6)
multiply()
```
"""

_AST_SAMPLE_CODE_EXECUTE = """
```
def multiply(a, b):
    return(5*6)
a = 5
b = 6

multiply(a, b)
```
"""


def test_python_repl() -> None:
    """Test functionality when globals/locals are not provided."""
    repl = PythonREPL()

    # Run a simple initial command.
    repl.run("foo = 1")
    assert repl.locals is not None
    assert repl.locals["foo"] == 1

    # Now run a command that accesses `foo` to make sure it still has it.
    repl.run("bar = foo * 2")
    assert repl.locals is not None
    assert repl.locals["bar"] == 2


def test_python_repl_no_previous_variables() -> None:
    """Test that it does not have access to variables created outside the scope."""
    foo = 3  # noqa: F841
    repl = PythonREPL()
    output = repl.run("print(foo)")
    assert output == """NameError("name 'foo' is not defined")"""


def test_python_repl_pass_in_locals() -> None:
    """Test functionality when passing in locals."""
    _locals = {"foo": 4}
    repl = PythonREPL(_locals=_locals)
    repl.run("bar = foo * 2")
    assert repl.locals is not None
    assert repl.locals["bar"] == 8


def test_functionality() -> None:
    """Test correct functionality."""
    chain = PythonREPL()
    code = "print(1 + 1)"
    output = chain.run(code)
    assert output == "2\n"


def test_functionality_multiline() -> None:
    """Test correct functionality for ChatGPT multiline commands."""
    chain = PythonREPL()
    tool = PythonREPLTool(python_repl=chain)
    output = tool.run(_SAMPLE_CODE)
    assert output == "30\n"


def test_python_ast_repl_multiline() -> None:
    """Test correct functionality for ChatGPT multiline commands."""
    if sys.version_info < (3, 9):
        pytest.skip("Python 3.9+ is required for this test")
    tool = PythonAstREPLTool()
    output = tool.run(_AST_SAMPLE_CODE)
    assert output == 30


def test_python_ast_repl_multi_statement() -> None:
    """Test correct functionality for ChatGPT multi statement commands."""
    if sys.version_info < (3, 9):
        pytest.skip("Python 3.9+ is required for this test")
    tool = PythonAstREPLTool()
    output = tool.run(_AST_SAMPLE_CODE_EXECUTE)
    assert output == 30


def test_function() -> None:
    """Test correct functionality."""
    chain = PythonREPL()
    code = "def add(a, b): " "    return a + b"
    output = chain.run(code)
    assert output == ""

    code = "print(add(1, 2))"
    output = chain.run(code)
    assert output == "3\n"
