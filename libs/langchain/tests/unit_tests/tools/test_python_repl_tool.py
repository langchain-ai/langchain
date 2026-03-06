from langchain.tools.python.tool import PythonAstREPLTool, PythonREPLTool
from langchain.utilities.python import PythonREPL


def test_python_repl_execution() -> None:
    """Test executing a simple python expression."""
    repl = PythonREPL()
    result = repl.run("print(1 + 1)")
    assert "2" in result


def test_python_repl_tool() -> None:
    """Test PythonREPLTool executes simple python expressions and captures stdout."""
    tool = PythonREPLTool()
    # Evaluating a simple addition and printing it
    result = tool.run("print(5 * 5)")
    assert "25" in result


def test_python_ast_repl_tool() -> None:
    """Test PythonAstREPLTool executes expressions and returns results natively."""
    tool = PythonAstREPLTool()
    # Given an AST-based tool, evaluating an expression returns the actual value
    result = tool.run("5 * 5")
    assert str(25) in str(result)
