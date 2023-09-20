# NOTE: ports are set to be different in each test so they don't collide
# as sockets might not be available immediately after container is stopped.
# It means that tests might also fail to run if port is not freed in time!
from langchain.utilities.python_docker_repl import PythonContainerREPL


def test_python_container_repl_can_be_started() -> None:
    repl = PythonContainerREPL(port=7120)
    assert repl is not None
    del repl


def test_python_container_repl_works():
    repl = PythonContainerREPL(port=7121)
    out1 = repl.exec("x = [1, 2, 3]")
    assert out1 == ""
    out2 = repl.eval("len(x)")
    assert out2 == "3"
    out3 = repl.exec("len(x)")
    assert out3 == ""
    out4 = repl.exec("print(len(x))")
    assert out4 == "3\n"

    err = repl.exec("print(x")
    assert "SyntaxError" in err

    err2 = repl.eval("print(x")
    assert "SyntaxError" in err2


def test_python_container_exec_code() -> None:
    repl = PythonContainerREPL(port=7122)
    code = """def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
print(fib(6))
"""
    out = repl.exec(code)
    assert out == "8\n"
    out2 = repl.exec("print(fib(5))")
    assert out2 == "5\n"


def test_python_container_ast_code() -> None:
    repl = PythonContainerREPL(port=7123)
    code = """def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
fib(6)
"""
    out = repl.eval(code)
    assert out == "8"
    out2 = repl.eval("fib(5)")
    assert out2 == "5"
