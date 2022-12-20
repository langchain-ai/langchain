"""Test functionality of Python REPL."""

from langchain.python import PythonREPL


def test_python_repl() -> None:
    """Test functionality when globals/locals are not provided."""
    repl = PythonREPL()

    # Run a simple initial command.
    repl.run("foo = 1")
    assert repl._locals["foo"] == 1

    # Now run a command that accesses `foo` to make sure it still has it.
    repl.run("bar = foo * 2")
    assert repl._locals["bar"] == 2


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
    assert repl._locals["bar"] == 8


def test_functionality() -> None:
    """Test correct functionality."""
    chain = PythonREPL()
    code = "print(1 + 1)"
    output = chain.run(code)
    assert output == "2\n"
