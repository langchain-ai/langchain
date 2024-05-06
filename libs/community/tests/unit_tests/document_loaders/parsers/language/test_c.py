import unittest

import pytest

from langchain_community.document_loaders.parsers.language.c import CSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestCSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """int main() {
    return 0;
}

struct S {
};

union U {
};

enum Evens {
    Two = 2,
    Four = 4
};"""

        self.expected_simplified_code = """// Code for: int main() {

// Code for: struct S {

// Code for: union U {

// Code for: enum Evens {"""

        self.expected_extracted_code = [
            "int main() {\n    return 0;\n}",
            "struct S {\n}",
            "union U {\n}",
            "enum Evens {\n    Two = 2,\n    Four = 4\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(CSegmenter("int a;").is_valid())
        self.assertFalse(CSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = CSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = CSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
