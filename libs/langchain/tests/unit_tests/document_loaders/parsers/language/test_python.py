import unittest

from langchain.document_loaders.parsers.language.python import PythonSegmenter


class TestPythonSegmenter(unittest.TestCase):
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
        segmenter = PythonSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = PythonSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
