import unittest

import pytest

from langchain_community.document_loaders.parsers.language.java import JavaSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestJavaSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """class Hello
{
    public static void main(String[] args)
    {
        System.out.println("Hello, world.");
    }
}

interface Human
{
    void breathe();
}

enum Tens
{
    TEN,
    TWENTY
}
"""

        self.expected_simplified_code = """// Code for: class Hello

// Code for: interface Human

// Code for: enum Tens"""

        self.expected_extracted_code = [
            "class Hello\n{\n    "
            "public static void main(String[] args)\n    {\n        "
            'System.out.println("Hello, world.");\n    }\n}',
            "interface Human\n{\n    void breathe();\n}",
            "enum Tens\n{\n    TEN,\n    TWENTY\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(JavaSegmenter("int a;").is_valid())
        self.assertFalse(JavaSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = JavaSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = JavaSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
