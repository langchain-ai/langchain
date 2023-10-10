import unittest

from langchain.document_loaders.parsers.language.cpp import CPPSegmenter


# @pytest.mark.requires("tree-sitter")
class TestCPPSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """int foo() {
    return 1;
}

class T {
    auto bar() const -> int;
    template<class U>
    void baz(U) {
    }
};

auto T::bar() const -> int {
    return 1;
}"""

        self.expected_simplified_code = """// Code for: int foo() {

// Code for: class T {

// Code for: auto T::bar() const -> int {"""

        self.expected_extracted_code = [
            "int foo() {\n    return 1;\n}",
            "class T{\n    auto bar() const -> int;\n    template<class U>\n    void baz(U) {\n    }\n}",
            "auto T::bar() const -> int {\n    return 1;\n}"
        ]

    def test_extract_functions_classes(self) -> None:
        segmenter = CPPSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = CPPSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
