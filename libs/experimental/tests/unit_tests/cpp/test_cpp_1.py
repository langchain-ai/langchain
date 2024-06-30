import unittest

from langchain_experimental.tools.cpp.tool import CppSubprocessTool, sanitize_cpp_input


class TestCppTool(unittest.TestCase):
    def test_sanitize_cpp_input(self) -> None:
        sanitized_code = sanitize_cpp_input("   `int main() { return 0; }`   ")
        self.assertEqual(sanitized_code, "int main() { return 0; }")

    def test_sanitize_cpp_input_no_backticks(self) -> None:
        sanitized_code = sanitize_cpp_input("   int main() { return 0; }   ")
        self.assertEqual(sanitized_code, "int main() { return 0; }")

    def test_sanitize_cpp_input_empty_string(self) -> None:
        sanitized_code = sanitize_cpp_input("      ")
        self.assertEqual(sanitized_code, "")

    def test_cpp_tool_successful_compilation_and_execution(self) -> None:
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        int main() {
            std::cout << "Hello, world!" << std::endl;
            return 0;
        }
        """
        output = tool._run(code)
        self.assertEqual(output.strip(), "Hello, world!")

    def test_cpp_tool_successful_complex_code(self) -> None:
        tool = CppSubprocessTool()
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
        self.assertEqual(output.strip(), "1 2 3 4 5")

    def test_cpp_tool_large_output(self) -> None:
        tool = CppSubprocessTool()
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
        self.assertTrue(len(output.strip()) > 10000)

    def test_cpp_tool_compilation_error(self) -> None:
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        int main() {
            std::cout << "Hello, world!" << std::endl;
            return 0
        }
        """
        output = tool._run(code)
        self.assertIn("Compilation failed:", output)

    def test_cpp_tool_execution_error(self) -> None:
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        int main() {
            int a = 1 / 0;
            std::cout << a << std::endl;
            return 0;
        }
        """
        output = tool._run(code)
        self.assertIn("Execution failed:", output)

    def test_cpp_tool_with_cpu_limit(self) -> None:
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        #include <unistd.h>
        int main() {
            while (true) { }
            return 0;
        }
        """
        output = tool._run(code, cpu_limit=1)
        self.assertIn("Execution failed:", output)

    def test_cpp_tool_with_c_language(self) -> None:
        tool = CppSubprocessTool()
        code = """
        #include <stdio.h>
        int main() {
            printf("Hello, C language!\\n");
            return 0;
        }
        """
        output = tool._run(code, language="c")
        self.assertEqual(output.strip(), "Hello, C language!")

    def test_cpp_tool_invalid_standard(self) -> None:
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        int main() {
            std::cout << "Invalid standard test" << std::endl;
            return 0;
        }
        """
        output = tool._run(code, std="invalid_standard")
        self.assertIn("Compilation failed:", output)


if __name__ == "__main__":
    unittest.main()
