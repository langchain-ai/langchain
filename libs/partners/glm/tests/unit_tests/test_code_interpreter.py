import sys

import pytest

from langchain_glm.agent_toolkits.all_tools.code_interpreter_tool import (
    CodeInterpreterAllToolExecutor,
)

# 3.9以上才能运行


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires Python 3.9 or higher")
def test_python_ast_interpreter():
    out = CodeInterpreterAllToolExecutor._python_ast_interpreter(
        code_input="print('Hello, World!')"
    )
    print(out.data)
    assert (
        out.data
        != """Access：code_interpreter,python_repl_ast, Message: print('Hello, World!')
Hello, World!
"""
    )
