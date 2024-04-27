import unittest
from langchain_experimental.utilities import PythonREPL

class TestSanitizeInput(unittest.TestCase):

    def test_whitespace_removal(self):
        query = "   print('Hello, world!')   "
        sanitized_query = PythonREPL.sanitize_input(query)
        self.assertEqual(sanitized_query, "print('Hello, world!')")

    def test_python_removal(self):
        query = "python   print('Hello, world!')   "
        sanitized_query = PythonREPL.sanitize_input(query)
        self.assertEqual(sanitized_query, "print('Hello, world!')")

    def test_backtick_removal(self):
        query = "`print('Hello, world!')`"
        sanitized_query = PythonREPL.sanitize_input(query)
        self.assertEqual(sanitized_query, "print('Hello, world!')")

    def test_combined_removal(self):
        query = "  `python  print('Hello, world!')`  "
        sanitized_query = PythonREPL.sanitize_input(query)
        self.assertEqual(sanitized_query, "print('Hello, world!')")

    def test_mixed_case_removal(self):
        query = "  pYtHoN   print('Hello, world!')  "
        sanitized_query = PythonREPL.sanitize_input(query)
        self.assertEqual(sanitized_query, "print('Hello, world!')")

if __name__ == '__main__':
    unittest.main()