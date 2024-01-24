import unittest

from langchain_community.document_loaders.parsers.language.python import PythonSegmenter


class TestPythonSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """import os

@first_decorator
@second_decorator
def hello(text):
    print(text)

def multiple_hello(
    text: str,
    count: int = 1,
):
    for _ in range(count):
        print(text)

class Alpha:
    def __init__(self):
        self.a = 1

@cls_decorator
class Beta:
    def __init__(self):
        self.b = 1

class AlphaBeta(
    Alpha,
    Beta,
):
    def __init__(self):
        super().__init__()

hello("Hello!")"""

        self.expected_simplified_code = """import os

# Code for: def hello(text):

# Code for: def multiple_hello(text: str, count: int=1):

# Code for: class Alpha:

# Code for: class Beta:

# Code for: class AlphaBeta(Alpha, Beta):

hello("Hello!")"""

        self.expected_extracted_code = [
            "@first_decorator\n" "@second_decorator\n" "def hello(text):\n" "    print(text)",
            "def multiple_hello(\n" "    text: str,\n" "    count: int = 1,\n" "):\n" "    for _ in range(count):\n" "        print(text)",
            "class Alpha:\n" "    def __init__(self):\n" "        self.a = 1",
            "@cls_decorator\n" "class Beta:\n" "    def __init__(self):\n" "        self.b = 1",
            "class AlphaBeta(\n" "    Alpha,\n" "    Beta,\n" "):\n" "    def __init__(self):\n" "        super().__init__()",
        ]

    def test_extract_functions_classes(self) -> None:
        segmenter = PythonSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = PythonSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
