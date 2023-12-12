import unittest

import pytest

from langchain_community.document_loaders.parsers.language.csharp import CSharpSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestCSharpSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """namespace World
{
}

class Hello
{
    static void Main(string []args)
    {
        System.Console.WriteLine("Hello, world.");
    }
}

interface Human
{
    void breathe();
}

enum Tens
{
    Ten = 10,
    Twenty = 20
}

struct T
{
}

record Person(string FirstName, string LastName, string Id)
{
    internal string Id { get; init; } = Id;
}"""

        self.expected_simplified_code = """// Code for: namespace World

// Code for: class Hello

// Code for: interface Human

// Code for: enum Tens

// Code for: struct T

// Code for: record Person(string FirstName, string LastName, string Id)"""

        self.expected_extracted_code = [
            "namespace World\n{\n}",
            "class Hello\n{\n    static void Main(string []args)\n    {\n        "
            'System.Console.WriteLine("Hello, world.");\n    }\n}',
            "interface Human\n{\n    void breathe();\n}",
            "enum Tens\n{\n    Ten = 10,\n    Twenty = 20\n}",
            "struct T\n{\n}",
            "record Person(string FirstName, string LastName, string Id)\n{\n    "
            "internal string Id { get; init; } = Id;\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(CSharpSegmenter("int a;").is_valid())
        self.assertFalse(CSharpSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = CSharpSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = CSharpSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
