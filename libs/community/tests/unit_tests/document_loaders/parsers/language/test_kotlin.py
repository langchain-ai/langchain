import unittest

import pytest

from langchain_community.document_loaders.parsers.language.kotlin import KotlinSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestKotlinSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """fun foo(a: Int): Int {
    return a
}

class T {
    var a: Int = 0
    var b: Boolean = false
    var c: String = ""
}

interface S {
    fun bar(): Double
}

enum class P {
    A,
    B,
    C
}
"""

        self.expected_simplified_code = """// Code for: fun foo(a: Int): Int {

// Code for: class T {

// Code for: interface S {

// Code for: enum class P {"""

        self.expected_extracted_code = [
            "fun foo(a: Int): Int {\n    return a\n}",
            "class T {\n    var a: Int = 0\n    var b: Boolean = false\n    "
            'var c: String = ""\n}',
            "interface S {\n    fun bar(): Double\n}",
            "enum class P {\n    A,\n    B,\n    C\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(KotlinSegmenter("val a: Int = 5").is_valid())
        self.assertFalse(KotlinSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = KotlinSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = KotlinSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
