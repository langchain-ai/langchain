import unittest

import pytest

from langchain_community.document_loaders.parsers.language.go import GoSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestGoSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """func foo(a int) int {
    return a;
}

type T struct {
    a int
    b bool
    c string
}

type S interface {
    bar() float64
}
"""

        self.expected_simplified_code = """// Code for: func foo(a int) int {

// Code for: type T struct {

// Code for: type S interface {"""

        self.expected_extracted_code = [
            "func foo(a int) int {\n    return a;\n}",
            "type T struct {\n    a int\n    b bool\n    c string\n}",
            "type S interface {\n    bar() float64\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(GoSegmenter("var a int;").is_valid())
        self.assertFalse(GoSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = GoSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = GoSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
