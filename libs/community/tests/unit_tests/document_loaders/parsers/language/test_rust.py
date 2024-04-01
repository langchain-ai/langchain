import unittest

import pytest

from langchain_community.document_loaders.parsers.language.rust import RustSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestRustSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """fn foo() -> i32 {
    return 1;
}

struct T {
    a: i32,
    b: bool,
    c: String
}

trait S {
    fn bar() -> Self
}
"""

        self.expected_simplified_code = """// Code for: fn foo() -> i32 {

// Code for: struct T {

// Code for: trait S {"""

        self.expected_extracted_code = [
            "fn foo() -> i32 {\n    return 1;\n}",
            "struct T {\n    a: i32,\n    b: bool,\n    c: String\n}",
            "trait S {\n    fn bar() -> Self\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(RustSegmenter("let a: i32;").is_valid())
        self.assertFalse(RustSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = RustSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = RustSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
