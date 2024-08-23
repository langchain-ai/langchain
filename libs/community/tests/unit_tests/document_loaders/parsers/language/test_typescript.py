import unittest

import pytest

from langchain_community.document_loaders.parsers.language.typescript import (
    TypeScriptSegmenter,
)


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestTypeScriptSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """function foo(): number
{
    return 1;
}

class Autumn
{
    leafCount = 45;
    reduceTemperature(desiredTemperature: number): number {
        return desiredTemperature * 0.6;
    }
}

interface Season
{
    change(): void;
}

enum Colors
{
    Green = 'green',
    Red = 'red',
}
"""

        self.expected_simplified_code = """// Code for: function foo(): number

// Code for: class Autumn

// Code for: interface Season

// Code for: enum Colors"""

        self.expected_extracted_code = [
            "function foo(): number\n{\n    return 1;\n}",
            "class Autumn\n{\n    leafCount = 45;\n    "
            "reduceTemperature(desiredTemperature: number): number {\n        "
            "return desiredTemperature * 0.6;\n    }\n}",
            "interface Season\n{\n    change(): void;\n}",
            "enum Colors\n{\n    Green = 'green',\n    Red = 'red',\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(TypeScriptSegmenter("let a;").is_valid())
        self.assertFalse(TypeScriptSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = TypeScriptSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = TypeScriptSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
