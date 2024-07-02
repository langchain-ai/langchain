import pytest

from langchain_experimental.tools.cpp.tool import CppSubprocessTool, sanitize_cpp_input


def test_sanitize_cpp_input() -> None:
    sanitized_code = sanitize_cpp_input("   `int main() { return 0; }`   ")
    assert sanitized_code == "int main() { return 0; }"


def test_sanitize_cpp_input_no_backticks() -> None:
    sanitized_code = sanitize_cpp_input("   int main() { return 0; }   ")
    assert sanitized_code == "int main() { return 0; }"


def test_sanitize_cpp_input_empty_string() -> None:
    sanitized_code = sanitize_cpp_input("      ")
    assert sanitized_code == ""


def test_cpp_tool_successful_compilation_and_execution() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    int main() {
        std::cout << "Hello, world!" << std::endl;
        return 0;
    }
    """
    output = tool._run(code)
    assert output.strip() == "Hello, world!"


def test_cpp_tool_successful_complex_code() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    #include <vector>
    int main() {
        std::vector<int> v = {1, 2, 3, 4, 5};
        for(int i : v) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        return 0;
    }
    """
    output = tool._run(code)
    assert output.strip() == "1 2 3 4 5"


def test_cpp_tool_large_output() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    int main() {
        for(int i = 0; i < 10000; ++i) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        return 0;
    }
    """
    output = tool._run(code)
    assert len(output.strip()) > 10000


def test_cpp_tool_compilation_error() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    int main() {
        std::cout << "Hello, world!" << std::endl;
        return 0
    }
    """
    output = tool._run(code)
    assert "Compilation failed:" in output


def test_cpp_tool_execution_error() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    int main() {
        int a = 1 / 0;
        std::cout << a << std::endl;
        return 0;
    }
    """
    output = tool._run(code)
    assert "Execution failed:" in output


def test_cpp_tool_with_c_language() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <stdio.h>
    int main() {
        printf("Hello, C language!\\n");
        return 0;
    }
    """
    output = tool._run(code, language="c")
    assert output.strip() == "Hello, C language!"


def test_cpp_tool_with_invalid_language() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    int main() {
        std::cout << "Hello, world!" << std::endl;
        return 0;
    }
    """
    result = tool._run(code, language="java")
    assert "Invalid language specified" in result


def test_cpp_tool_without_dangerous_code_permission() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=False)
    code = """
    #include <iostream>
    int main() {
        std::cout << "Hello, world!" << std::endl;
        return 0;
    }
    """
    try:
        output = tool._run(code)
        output
    except PermissionError as e:
        assert "Execution of C/C++ code is disabled by default" in str(e)
    else:
        assert False, "Expected PermissionError for disallowed dangerous code"


def test_cpp_tool_with_nonexistent_file() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    #include <fstream>
    int main() {
        std::ifstream file("nonexistent.txt");
        if (!file.is_open()) {
            std::cerr << "File not found" << std::endl;
            return 1;
        }
        return 0;
    }
    """
    output = tool._run(code)
    assert "Execution failed:" in output
    assert "File not found" in output


def test_cpp_tool_with_cpu_limit() -> None:
    tool = CppSubprocessTool(allow_dangerous_code=True)
    code = """
    #include <iostream>
    #include <unistd.h>
    int main() {
        usleep(500000); // Sleep for 0.5 seconds
        std::cout << "Execution within CPU limit" << std::endl;
        return 0;
    }
    """
    output = tool._run(code, cpu_limit=1)
    assert "Execution within CPU limit" in output
    assert "Execution failed:" not in output


if __name__ == "__main__":
    pytest.main()
