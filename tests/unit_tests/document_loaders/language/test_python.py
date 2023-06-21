import unittest

from langchain.document_loaders.language.python import PythonParser


class TestPythonParser(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """import os

def hello(text):
    print(text)

class Simple:
    def __init__(self):
        self.a = 1

hello("Hello!")"""

        self.expected_simplified_code = """import os

# Code for: def hello(text):

# Code for: class Simple:

hello("Hello!")"""

        self.expected_extracted_code = [
            "def hello(text):\n" "    print(text)",
            "class Simple:\n" "    def __init__(self):\n" "        self.a = 1",
        ]

    def test_extract_functions_classes(self) -> None:
        parser = PythonParser(self.example_code)
        extracted_code = parser.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        parser = PythonParser(self.example_code)
        simplified_code = parser.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
