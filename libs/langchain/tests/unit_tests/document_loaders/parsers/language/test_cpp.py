import unittest

import pytest

from langchain.document_loaders.parsers.language.cpp import CPPSegmenter


# @pytest.mark.requires("tree-sitter")
class TestCPPSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """int T::foo() { return 1; }
int T::foo() const { return 0; }"""

    def test_is_valid(self) -> None:
        segmenter = CPPSegmenter(self.example_code)
        raise RuntimeError("is_valid? " + str(segmenter.is_valid()))

    # def test_extract_functions_classes(self) -> None:
    #     segmenter = JavaScriptSegmenter(self.example_code)
    #     extracted_code = segmenter.extract_functions_classes()
    #     self.assertEqual(extracted_code, self.expected_extracted_code)

    # def test_simplify_code(self) -> None:
    #     segmenter = JavaScriptSegmenter(self.example_code)
    #     simplified_code = segmenter.simplify_code()
    #     self.assertEqual(simplified_code, self.expected_simplified_code)
