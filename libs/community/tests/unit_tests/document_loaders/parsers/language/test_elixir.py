import unittest

import pytest

from langchain_community.document_loaders.parsers.language.elixir import ElixirSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestElixirSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """@doc "some comment"
def foo do
  i = 0
end

defmodule M do
  def hi do
    i = 2
  end

  defp wave do
    :ok
  end
end"""

        self.expected_simplified_code = """# Code for: @doc "some comment"
# Code for: def foo do

# Code for: defmodule M do"""

        self.expected_extracted_code = [
            '@doc "some comment"',
            "def foo do\n  i = 0\nend",
            "defmodule M do\n"
            "  def hi do\n"
            "    i = 2\n"
            "  end\n\n"
            "  defp wave do\n"
            "    :ok\n"
            "  end\n"
            "end",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(ElixirSegmenter("def a do; end").is_valid())
        self.assertFalse(ElixirSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = ElixirSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(len(extracted_code), 3)
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = ElixirSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
