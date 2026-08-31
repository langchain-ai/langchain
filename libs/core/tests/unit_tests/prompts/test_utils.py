"""Test functionality related to prompt utils."""

from langchain_core.documents import Document
from langchain_core.example_selectors import sorted_values
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import format_document


def test_sorted_vals() -> None:
    """Test sorted values from dictionary."""
    test_dict = {"key2": "val2", "key1": "val1"}
    expected_response = ["val1", "val2"]
    assert sorted_values(test_dict) == expected_response


def test_format_document_metadata_does_not_override_page_content() -> None:
    """Regression test for metadata keys not overriding page_content.

    Metadata keys named 'page_content' must not override
    the actual Document.page_content in format_document.
    See: https://github.com/langchain-ai/langchain/issues/40073
    """
    doc = Document(
        page_content="actual content",
        metadata={"page_content": "metadata value", "extra": "data"},
    )
    prompt = PromptTemplate.from_template("{page_content}")
    result = format_document(doc, prompt)
    assert result == "actual content"


def test_format_document_metadata_does_not_override_named_keys() -> None:
    """Metadata keys should not override any first-class Document attributes."""
    doc = Document(
        page_content="real content",
        metadata={"page_content": "fake", "extra": "data"},
    )
    prompt = PromptTemplate.from_template("{page_content} - {extra}")
    result = format_document(doc, prompt)
    assert result == "real content - data"
