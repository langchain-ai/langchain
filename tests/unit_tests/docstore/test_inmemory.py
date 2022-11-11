"""Test in memory docstore."""

from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore


def test_document_found() -> None:
    """Test document found."""
    _list = [Document(page_content="bar")]
    docstore = InMemoryDocstore(_list)
    output = docstore.search(0)
    assert isinstance(output, Document)
    assert output.page_content == "bar"


def test_document_not_found() -> None:
    """Test when document is not found."""
    _list = [Document(page_content="bar")]
    docstore = InMemoryDocstore(_list)
    output = docstore.search(0)
    assert output == "ID bar not found."
