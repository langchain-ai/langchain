import numpy as np

from langchain.tools.python.tool import PythonAstREPLTool, PythonREPLTool


def test_python_ast_repl_return():
    program = """
import numpy as np
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)
int(dot_product)
    """
    tool = PythonAstREPLTool()
    assert tool.run(program) == 32


def test_python_ast_repl_print():
    program = """
string = "racecar"
if string == string[::-1]:
    print(string, "is a palindrome")
else:
    print(string, "is not a palindrome")"""
    tool = PythonAstREPLTool()
    assert tool.run(program) == "racecar is a palindrome\n"


def test_python_ast_repl_raise_exception():
    data = {"Name": ["John", "Alice"], "Age": [30, 25]}
    program = """
import pandas as pd
df = pd.DataFrame(data)
df['Gender']
    """
    tool = PythonAstREPLTool(locals={"data": data})
    expected_outputs = (
        "KeyError: 'Gender'",
        "ModuleNotFoundError: No module named 'pandas'",
    )
    assert tool.run(program) in expected_outputs


def test_python_ast_repl_one_line_print():
    program = 'print("The square of {} is {:.2f}".format(3, 3**2))'
    tool = PythonAstREPLTool()
    assert tool.run(program) == "The square of 3 is 9.00\n"


def test_python_ast_repl_one_line_return():
    arr = np.array([1, 2, 3, 4, 5])
    tool = PythonAstREPLTool(locals={"arr": arr})
    program = "(arr**2).sum()   # Returns sum of squares"
    assert tool.run(program) == 55


def test_python_ast_repl_one_line_exception():
    program = "[1, 2, 3][4]"
    tool = PythonAstREPLTool()
    assert tool.run(program) == "IndexError: list index out of range"


def test_python_repl_print():
    program = """
import numpy as np
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)
print("The dot product is {:d}.".format(dot_product))
    """
    tool = PythonREPLTool()
    assert tool.run(program) == "The dot product is 32.\n"
