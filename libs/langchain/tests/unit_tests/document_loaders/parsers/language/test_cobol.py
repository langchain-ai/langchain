import unittest

from langchain.document_loaders.parsers.language.cobol import CobolSegmenter


class TestCobolSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. SampleProgram.
DATA DIVISION.
WORKING-STORAGE SECTION.
01  SAMPLE-VAR         PIC X(20)   VALUE 'Sample Value'.

PROCEDURE DIVISION.
A000-INITIALIZE-PARA.
    DISPLAY 'Initialization Paragraph'.
    MOVE 'New Value' TO SAMPLE-VAR.

A100-PROCESS-PARA.
    DISPLAY SAMPLE-VAR.
    STOP RUN.
"""
        self.expected_simplified_code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. SampleProgram.
DATA DIVISION.
WORKING-STORAGE SECTION.
* OMITTED CODE *
PROCEDURE DIVISION.
A000-INITIALIZE-PARA.
* OMITTED CODE *
A100-PROCESS-PARA.
* OMITTED CODE *
"""

        self.expected_extracted_code = [
            "A000-INITIALIZE-PARA.\n    DISPLAY 'Initialization Paragraph'.\n    MOVE 'New Value' TO SAMPLE-VAR.",  # noqa: E501
            "A100-PROCESS-PARA.\n    DISPLAY SAMPLE-VAR.\n    STOP RUN.",
        ]

    def test_extract_functions_classes(self) -> None:
        segmenter = CobolSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = CobolSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code.strip(), self.expected_simplified_code.strip())
