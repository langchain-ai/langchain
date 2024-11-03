import unittest

import pytest

from langchain_community.document_loaders.parsers.language.perl import PerlSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestPerlSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """sub Hello {
  print "Hello, World!";
}

sub new {
  my $class = shift;
  my $self = {};
  bless $self, $class;
  return $self;
}"""

        self.expected_simplified_code = """# Code for: sub Hello {

# Code for: sub new {"""

        self.expected_extracted_code = [
            'sub Hello {\n  print "Hello, World!";\n}',
            "sub new {\n  my $class = shift;\n  my $self = {};\n  "
            "bless $self, $class;\n  return $self;\n}",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(PerlSegmenter("$age = 25;").is_valid())
        self.assertFalse(PerlSegmenter("a b c 1 2 3").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = PerlSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = PerlSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
