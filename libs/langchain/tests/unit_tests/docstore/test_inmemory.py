"""Test in memory docstore."""
import pytest

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


def test_adding_document() -> None:
    """Test that documents are added correctly."""
    _dict = {"foo": Document(page_content="bar")}
    docstore = InMemoryDocstore(_dict)
    new_dict = {"bar": Document(page_content="foo")}
    docstore.add(new_dict)

    # Test that you can find new document.
    foo_output = docstore.search("bar")
    assert isinstance(foo_output, Document)
    assert foo_output.page_content == "foo"

    # Test that old document is the same.
    bar_output = docstore.search("foo")
    assert isinstance(bar_output, Document)
    assert bar_output.page_content == "bar"


def test_adding_document_already_exists() -> None:
    """Test that error is raised if document id already exists."""
    _dict = {"foo": Document(page_content="bar")}
    docstore = InMemoryDocstore(_dict)
    new_dict = {"foo": Document(page_content="foo")}

    # Test that error is raised.
    with pytest.raises(ValueError):
        docstore.add(new_dict)

    # Test that old document is the same.
    bar_output = docstore.search("foo")
    assert isinstance(bar_output, Document)
    assert bar_output.page_content == "bar"


def test_default_dict_value_in_constructor() -> None:
    """Test proper functioning if no _dict is provided to the constructor."""
    docstore = InMemoryDocstore()
    docstore.add({"foo": Document(page_content="bar")})
    output = docstore.search("foo")
    assert isinstance(output, Document)
    assert output.page_content == "bar"
