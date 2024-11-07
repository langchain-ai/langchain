import pytest
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.base import (
    Node,
    _documents_to_nodes,
    _texts_to_nodes,
)
from langchain_community.graph_vectorstores.links import Link


def test_texts_to_nodes() -> None:
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"])) == [
        Node(id="a", metadata={"a": "b"}, text="a"),
        Node(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], None, ["a", "b"])) == [
        Node(id="a", metadata={}, text="a"),
        Node(id="b", metadata={}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], None)) == [
        Node(metadata={"a": "b"}, text="a"),
        Node(metadata={"c": "d"}, text="b"),
    ]
    assert list(
        _texts_to_nodes(
            ["a"],
            [{"links": {Link.incoming(kind="hyperlink", tag="http://b")}}],
            None,
        )
    ) == [Node(links=[Link.incoming(kind="hyperlink", tag="http://b")], text="a")]
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
        Document(
            id="a",
            page_content="some text a",
            metadata={"links": [Link.incoming(kind="hyperlink", tag="http://b")]},
        ),
        Document(id="b", page_content="some text b", metadata={"c": "d"}),
    ]
    assert list(_documents_to_nodes(documents)) == [
        Node(
            id="a",
            metadata={},
            links=[Link.incoming(kind="hyperlink", tag="http://b")],
            text="some text a",
        ),
        Node(id="b", metadata={"c": "d"}, text="some text b"),
    ]
