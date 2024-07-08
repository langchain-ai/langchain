import unittest

import pytest

from langchain_community.document_loaders.parsers.language.scala import ScalaSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestScalaSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """def foo() {
    return 1
}

object T {
    def baz() {
        val x = 1
    }
}

class S() {

}

trait T {
    def P(x: Any): Boolean
}"""

        self.expected_simplified_code = """// Code for: def foo() {

// Code for: object T {

// Code for: class S() {

// Code for: trait T {"""

        self.expected_extracted_code = [
            "def foo() {\n    return 1\n}",
            "object T {\n    def baz() {\n        val x = 1\n    }\n}",
            "class S() {\n\n}",
            "trait T {\n    def P(x: Any): Boolean\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertFalse(ScalaSegmenter("val x").is_valid())
        self.assertFalse(ScalaSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = ScalaSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = ScalaSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
