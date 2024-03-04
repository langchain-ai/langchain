import unittest

import pytest

from langchain_community.document_loaders.parsers.language.ruby import RubySegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestRubySegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """def foo
  i = 0
end

module M
  def hi
    i = 2
  end
end

class T
  def bar
    j = 1
  end
end"""

        self.expected_simplified_code = """# Code for: def foo

# Code for: module M

# Code for: class T"""

        self.expected_extracted_code = [
            "def foo\n  i = 0\nend",
            "module M\n  def hi\n    i = 2\n  end\nend",
            "class T\n  def bar\n    j = 1\n  end\nend",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(RubySegmenter("def a; end").is_valid())
        self.assertFalse(RubySegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = RubySegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = RubySegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
