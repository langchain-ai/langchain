"""Test Python REPL Tools."""
import sys

import numpy as np
import pytest

from langchain_experimental.tools.python.tool import (
    PythonAstREPLTool,
    PythonREPLTool,
    sanitize_input,
)


def test_python_repl_tool_single_input() -> None:
    """Test that the python REPL tool works with a single input."""
    tool = PythonREPLTool()
    assert tool.is_single_input
    assert int(tool.run("print(1 + 1)").strip()) == 2


def test_python_repl_print() -> None:
    program = """
import numpy as np
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)
print("The dot product is {:d}.".format(dot_product))
    """
    tool = PythonREPLTool()
    assert tool.run(program) == "The dot product is 32.\n"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_tool_single_input() -> None:
    """Test that the python REPL tool works with a single input."""
    tool = PythonAstREPLTool()
    assert tool.is_single_input
    assert tool.run("1 + 1") == 2


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_return() -> None:
    program = """
```
import numpy as np
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)
int(dot_product)
```
    """
    tool = PythonAstREPLTool()
    assert tool.run(program) == 32

    program = """
```python
import numpy as np
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)
int(dot_product)
```
    """
    assert tool.run(program) == 32


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_print() -> None:
    program = """python
string = "racecar"
if string == string[::-1]:
    print(string, "is a palindrome")
else:
    print(string, "is not a palindrome")"""
    tool = PythonAstREPLTool()
    assert tool.run(program) == "racecar is a palindrome\n"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_repl_print_python_backticks() -> None:
    program = "`print('`python` is a great language.')`"
    tool = PythonAstREPLTool()
    assert tool.run(program) == "`python` is a great language.\n"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_raise_exception() -> None:
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


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_one_line_print() -> None:
    program = 'print("The square of {} is {:.2f}".format(3, 3**2))'
    tool = PythonAstREPLTool()
    assert tool.run(program) == "The square of 3 is 9.00\n"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_one_line_return() -> None:
    arr = np.array([1, 2, 3, 4, 5])
    tool = PythonAstREPLTool(locals={"arr": arr})
    program = "`(arr**2).sum()   # Returns sum of squares`"
    assert tool.run(program) == 55


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_python_ast_repl_one_line_exception() -> None:
    program = "[1, 2, 3][4]"
    tool = PythonAstREPLTool()
    assert tool.run(program) == "IndexError: list index out of range"


def test_sanitize_input() -> None:
    query = """
    ```
        p = 5
    ```
    """
    expected = "p = 5"
    actual = sanitize_input(query)
    assert expected == actual

    query = """
       ```python
        p = 5
    ```
    """
    expected = "p = 5"
    actual = sanitize_input(query)
    assert expected == actual

    query = """
    p = 5
    """
    expected = "p = 5"
    actual = sanitize_input(query)
    assert expected == actual
