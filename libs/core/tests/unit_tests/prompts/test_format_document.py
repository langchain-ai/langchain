"""Tests for format_document and _get_document_info."""

import pytest

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import _get_document_info, format_document


def test_metadata_page_content_key_does_not_override_real_content() -> None:
    """Regression: a metadata field named 'page_content' must not replace the
    document's actual page_content in the formatted output."""
    doc = Document(
        page_content="real document text",
        metadata={"page_content": "metadata impostor", "source": "test.pdf"},
    )
    prompt = PromptTemplate.from_template("{page_content}")
    result = format_document(doc, prompt)
    assert result == "real document text"


def test_get_document_info_preserves_page_content_over_metadata() -> None:
    """_get_document_info must return the document's page_content, not a
    same-named metadata field."""
    doc = Document(
        page_content="the real content",
        metadata={"page_content": "shadowed", "author": "alice"},
    )
    prompt = PromptTemplate.from_template("{page_content} by {author}")
    info = _get_document_info(doc, prompt)
    assert info["page_content"] == "the real content"
    assert info["author"] == "alice"


def test_format_document_normal_metadata() -> None:
    """format_document works normally when metadata does not shadow page_content."""
    doc = Document(
        page_content="hello world",
        metadata={"source": "s3://bucket/key", "page": "42"},
    )
    prompt = PromptTemplate.from_template("Content: {page_content}\nPage: {page}")
    result = format_document(doc, prompt)
    assert result == "Content: hello world\nPage: 42"
