import unittest

from langchain.document_loaders.parsers.language.cobol import CobolSegmenter


class TestCobolSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. HelloWorld.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 GREETING           PIC X(12)   VALUE 'Hello, World!'.
PROCEDURE DIVISION.
DISPLAY GREETING.
STOP RUN.
"""

        self.expected_simplified_code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. HelloWorld.
DATA DIVISION.
WORKING-STORAGE SECTION.
PROCEDURE DIVISION.
"""

        self.expected_extracted_code = [
            "PROGRAM-ID. HelloWorld.",
            "01 GREETING           PIC X(12)   VALUE 'Hello, World!'.",
            "DISPLAY GREETING.",
            "STOP RUN.",
        ]

    def test_extract_functions_classes(self) -> None:
        segmenter = CobolSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = CobolSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code.strip(), self.expected_simplified_code.strip())
