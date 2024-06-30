import unittest
from langchain_experimental.tools.cpp.tool import CppSubprocessTool, sanitize_cpp_input

class TestCppTool(unittest.TestCase):

    def test_sanitize_cpp_input(self):
        sanitized_code = sanitize_cpp_input("   `int main() { return 0; }`   ")
        print(f"Sanitized code: '{sanitized_code}'")
        self.assertEqual(sanitized_code, "int main() { return 0; }")

    def test_sanitize_cpp_input_no_backticks(self):
        sanitized_code = sanitize_cpp_input("   int main() { return 0; }   ")
        print(f"Sanitized code: '{sanitized_code}'")
        self.assertEqual(sanitized_code, "int main() { return 0; }")

    def test_sanitize_cpp_input_empty_string(self):
        sanitized_code = sanitize_cpp_input("      ")
        print(f"Sanitized code: '{sanitized_code}'")
        self.assertEqual(sanitized_code, "")

    def test_cpp_tool_successful_compilation_and_execution(self):
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        int main() {
            std::cout << "Hello, world!" << std::endl;
            return 0;
        }
        """
        output = tool._run(code)
        print(f"Output: '{output}'")
        self.assertEqual(output.strip(), "Hello, world!")

    def test_cpp_tool_successful_complex_code(self):
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
        print(f"Output: '{output}'")
        self.assertEqual(output.strip(), "1 2 3 4 5")

    def test_cpp_tool_large_output(self):
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
        print(f"Output length: {len(output.strip())}")
        self.assertTrue(len(output.strip()) > 10000)

    def test_cpp_tool_compilation_error(self):
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        int main() {
            std::cout << "Hello, world!" << std::endl;
            return 0
        }
        """
        output = tool._run(code)
        print(f"Output: '{output}'")
        self.assertIn("Compilation failed:", output)

    def test_cpp_tool_execution_error(self):
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
        print(f"Output: '{output}'")
        self.assertIn("Execution failed:", output)

    def test_cpp_tool_with_cpu_limit(self):
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
        print(f"Output: '{output}'")
        self.assertIn("Execution failed:", output)

    def test_cpp_tool_with_c_language(self):
        tool = CppSubprocessTool()
        code = """
        #include <stdio.h>
        int main() {
            printf("Hello, C language!\\n");
            return 0;
        }
        """
        output = tool._run(code, language="c")
        print(f"Output: '{output}'")
        self.assertEqual(output.strip(), "Hello, C language!")

    def test_cpp_tool_invalid_standard(self):
        tool = CppSubprocessTool()
        code = """
        #include <iostream>
        int main() {
            std::cout << "Invalid standard test" << std::endl;
            return 0;
        }
        """
        output = tool._run(code, std="invalid_standard")
        print(f"Output: '{output}'")
        self.assertIn("Compilation failed:", output)

if __name__ == '__main__':
    unittest.main()
