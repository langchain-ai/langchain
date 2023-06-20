import unittest
import pytest

from langchain.document_loaders.language.javascript import JavaScriptParser


@pytest.mark.requires("esprima")
class TestJavaScriptParser(unittest.TestCase):
    def setUp(self):
        self.example_code = """const os = require('os');

function hello(text) {
    console.log(text);
}

class Simple {
    constructor() {
        this.a = 1;
    }
}

hello("Hello!");"""

        self.expected_simplified_code = """const os = require('os');

// Code for: function hello(text) {

// Code for: class Simple {

hello("Hello!");"""

        self.expected_extracted_code = [
            "function hello(text) {\n    console.log(text);\n}",
            "class Simple {\n    constructor() {\n        this.a = 1;\n    }\n}",
        ]

    def test_extract_functions_classes(self):
        parser = JavaScriptParser(self.example_code)
        extracted_code = parser.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)
        assert False

    def test_simplify_code(self):
        parser = JavaScriptParser(self.example_code)
        simplified_code = parser.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
