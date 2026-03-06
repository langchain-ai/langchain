from langchain.tools.python.tool import PythonAstREPLTool, PythonREPLTool
from langchain.utilities.python import PythonREPL


def test_repl():
    repl = PythonREPL()
    res = repl.run("print('hello world')")
    assert "hello world" in res, f"Expected hello world, got: {res}"
    print("PythonREPL standalone tests passed!")


def test_tool():
    tool = PythonREPLTool()
    res = tool.run("print(5 * 5)")
    assert "25" in res, f"Expected 25, got: {res}"
    print("PythonREPLTool tests passed!")

    ast_tool = PythonAstREPLTool()
    res = ast_tool.run("5 * 5")
    assert res == 25 or "25" in str(res), f"Expected 25, got: {res}"
    print("PythonAstREPLTool tests passed!")


if __name__ == "__main__":
    test_repl()
    test_tool()
    print("ALL TESTS PASSED!")
