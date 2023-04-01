"""Test functionality of Python REPL."""

from langchain.python import PythonREPL
from langchain.tools.python.tool import PythonAstREPLTool


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
    assert output == "name 'foo' is not defined"


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


def test_function() -> None:
    """Test correct functionality."""
    chain = PythonREPL()
    code = "def add(a, b): " "    return a + b"
    output = chain.run(code)
    assert output == ""

    code = "print(add(1, 2))"
    output = chain.run(code)
    assert output == "3\n"

# TODO(jon-chuang): add new file and pytest skipif version < (3, 9, 0)?
def test_tool_object_output() -> None:
    """Test returning an object from an expression."""
    chain  = PythonAstREPLTool()
    code = "class A:\n  def __init__(self, x):\n    print(x); self.x = x\nA(1+122).x"
    output = chain.run(code)
    # Only return the final output
    assert output == "123"

def test_tool_stdout_output() -> None:
    """Test returning an object from an expression."""
    chain  = PythonAstREPLTool()
    code = "def f(x): print(x+1)\nf(123)"
    output = chain.run(code)
    assert output == "124\n"

def test_tool_throw_exception() -> None:
    """Test returning an object from an expression."""
    chain  = PythonAstREPLTool()
    code = "def f(x): raise ValueError(f'{x}')\nf(123)"
    output = chain.run(code)
    assert output == "ValueError: 123"

def test_tool_invalid_stmt() -> None:
    """Test returning an object from an expression."""
    chain  = PythonAstREPLTool()
    code = "x = 5"
    output = chain.run(code)
    assert output == "SyntaxError: invalid syntax (<string>, line 1)"