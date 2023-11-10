"""Integration tests for LLMEncodedChainExtractor."""
import pytest

from langchain.llms import OpenAI
from langchain.retrievers.document_compressors import LLMEncodedChainExtractor
from langchain.retrievers.document_compressors.encoded_chain_extract import (
    number_sequences,
)
from langchain.schema import Document


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


@pytest.mark.requires("spacy")
def test_llm_chain_extractor() -> None:
    texts = [
        "The Roman Empire followed the Roman Republic.",
        "I love chocolate chip cookies—my mother makes great cookies.",
        "The first Roman emperor was Caesar Augustus.",
        "Don't you just love Caesar salad?",
        "The Roman Empire collapsed in 476 AD after the fall of Rome.",
        "Let's go to Olive Garden!",
    ]
    doc = Document(page_content=" ".join(texts))
    compressor = LLMEncodedChainExtractor.from_llm(OpenAI())
    actual = compressor.compress_documents([doc], "Tell me about the Roman Empire")[
        0
    ].page_content
    expected_returned = [0, 2, 4]
    expected_not_returned = [1, 3, 5]
    assert all(texts[i] in actual for i in expected_returned)
    assert all(texts[i] not in actual for i in expected_not_returned)


@pytest.mark.requires("spacy")
def test_llm_chain_extractor_empty() -> None:
    texts = [
        "I love chocolate chip cookies—my mother makes great cookies.",
        "Don't you just love Caesar salad?",
        "Let's go to Olive Garden!",
    ]
    doc = Document(page_content=" ".join(texts))
    compressor = LLMEncodedChainExtractor.from_llm(OpenAI())
    actual = compressor.compress_documents([doc], "Tell me about the Roman Empire")
    assert len(actual) == 0
