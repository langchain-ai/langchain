import unittest

import pytest

from langchain.document_loaders.parsers.language.javascript import JavaScriptSegmenter


@pytest.mark.requires("esprima")
class TestJavaScriptSegmenter(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_extract_functions_classes(self) -> None:
        segmenter = JavaScriptSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = JavaScriptSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
