"""Test functionality of Python REPL."""

import pytest

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
    with pytest.raises(NameError):
        repl.run("print(foo)")


def test_python_repl_pass_in_locals() -> None:
    """Test functionality when passing in locals."""
    _locals = {"foo": 4}
    repl = PythonREPL(_locals=_locals)
    repl.run("bar = foo * 2")
    assert repl._locals["bar"] == 8
