"""Test in memory docstore."""

from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore


def test_document_found() -> None:
    """Test document found."""
    _dict = {"foo": Document(page_content="bar")}
    docstore = InMemoryDocstore(_dict)
    output = docstore.search("foo")
    assert isinstance(output, Document)
    assert output.page_content == "bar"


def test_document_not_found() -> None:
    """Test when document is not found."""
    _dict = {"foo": Document(page_content="bar")}
    docstore = InMemoryDocstore(_dict)
    output = docstore.search("bar")
    assert output == "ID bar not found."
