"""Unit tests for LLMEncodedChainExtractor."""
import pytest

from langchain.retrievers.document_compressors.encoded_chain_extract import (
    SequenceListParser,
    extract_numbered_sequences,
    number_sequences,
)
from langchain.schema.output_parser import OutputParserException


def test_parser() -> None:
    parser = SequenceListParser()
    assert parser.parse("1,2,3") == [1, 2, 3]
    assert parser.parse("1-3") == [1, 2, 3]
    assert parser.parse("1-3,5") == [1, 2, 3, 5]
    assert parser.parse("1-3,5-7") == [1, 2, 3, 5, 6, 7]
    with pytest.raises(OutputParserException):
        assert parser.parse("1,6,10-8")
    with pytest.raises(OutputParserException):
        assert parser.parse("1,6,a")


@pytest.mark.requires("spacy")
def test_number_sequences() -> None:
    assert number_sequences("foo") == "#|1|# foo"
    assert number_sequences("foo bar") == "#|1|# foo bar"
    assert number_sequences("foo\nbar") == "#|1|# foo\nbar"
    assert number_sequences("foo\n\nbar") == "#|1|# foo  \n\n  #|2|# bar"
    assert (
        number_sequences("foo\n\nbar\n\nbaz", 2)
        == "#|1|# foo  \n\n   bar  \n\n  #|2|# baz"
    )
    assert (
        number_sequences("foo\n\n\nbar\n\n\nbaz", 4)
        == "#|1|# foo  \n\n   bar  \n\n   baz"
    )


def test_extract_numbered_sequences() -> None:
    assert extract_numbered_sequences("#|1|# foo", []) == ""
    assert extract_numbered_sequences("#|1|# foo", [1]) == "foo"
    assert extract_numbered_sequences("#|1|# foo", [2]) == ""
    assert extract_numbered_sequences("#|1|# foo\n\n  #|2|# bar", [1]) == "foo"
    assert extract_numbered_sequences("#|1|# foo\n\n  #|2|# bar", [2]) == "bar"
