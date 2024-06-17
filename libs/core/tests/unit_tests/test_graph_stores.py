import pytest

from langchain_core.documents import Document
from langchain_core.graph_stores import TextNode, _documents_to_nodes, _texts_to_nodes


def test_texts_to_nodes() -> None:
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"])) == [
        TextNode(id="a", metadata={"a": "b"}, text="a"),
        TextNode(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], None, ["a", "b"])) == [
        TextNode(id="a", metadata={}, text="a"),
        TextNode(id="b", metadata={}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], None)) == [
        TextNode(id=None, metadata={"a": "b"}, text="a"),
        TextNode(id=None, metadata={"c": "d"}, text="b"),
    ]
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a", "b"], None, ["a"]))
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a", "b"], [{"a": "b"}], None))
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a"], [{"a": "b"}, {"c": "d"}], None))
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a"], None, ["a", "b"]))


def test_documents_to_nodes() -> None:
    documents = [
        Document(page_content="a", metadata={"a": "b"}),
        Document(page_content="b", metadata={"c": "d"}),
    ]
    assert list(_documents_to_nodes(documents, ["a", "b"])) == [
        TextNode(id="a", metadata={"a": "b"}, text="a"),
        TextNode(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_documents_to_nodes(documents, None)) == [
        TextNode(id=None, metadata={"a": "b"}, text="a"),
        TextNode(id=None, metadata={"c": "d"}, text="b"),
    ]
    with pytest.raises(ValueError):
        list(_documents_to_nodes(documents, ["a"]))
    with pytest.raises(ValueError):
        list(_documents_to_nodes(documents[1:], ["a", "b"]))
