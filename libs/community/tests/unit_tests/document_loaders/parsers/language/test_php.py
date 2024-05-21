import unittest

import pytest

from langchain_community.document_loaders.parsers.language.php import PHPSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestPHPSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """<?php
namespace foo;

class Hello {
    public function __construct() { }
}

function hello() {
    echo "Hello World!";
}

interface Human {
    public function breath();
}

trait Foo { }

enum Color
{
    case Red;
    case Blue;
}"""

        self.expected_simplified_code = """<?php
// Code for: namespace foo;

// Code for: class Hello {

// Code for: function hello() {

// Code for: interface Human {

// Code for: trait Foo { }

// Code for: enum Color"""

        self.expected_extracted_code = [
            "namespace foo;",
            "class Hello {\n    public function __construct() { }\n}",
            'function hello() {\n    echo "Hello World!";\n}',
            "interface Human {\n    public function breath();\n}",
            "trait Foo { }",
            "enum Color\n{\n    case Red;\n    case Blue;\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(PHPSegmenter("<?php $a = 0;").is_valid())
        self.assertFalse(PHPSegmenter("<?php a ?b}+ c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = PHPSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = PHPSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
