import pytest
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.base import (
    GraphStoreNode,
    _documents_to_nodes,
    _texts_to_nodes,
)
from langchain_community.graph_vectorstores.links import GraphStoreLink


def test_texts_to_nodes() -> None:
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"])) == [
        GraphStoreNode(id="a", metadata={"a": "b"}, text="a"),
        GraphStoreNode(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], None, ["a", "b"])) == [
        GraphStoreNode(id="a", metadata={}, text="a"),
        GraphStoreNode(id="b", metadata={}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], None)) == [
        GraphStoreNode(metadata={"a": "b"}, text="a"),
        GraphStoreNode(metadata={"c": "d"}, text="b"),
    ]
    assert list(
        _texts_to_nodes(
            ["a"],
            [{"links": {GraphStoreLink.incoming(kind="hyperlink", tag="http://b")}}],
            None,
        )
    ) == [
        GraphStoreNode(
            links=[GraphStoreLink.incoming(kind="hyperlink", tag="http://b")], text="a"
        )
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
        Document(
            id="a",
            page_content="some text a",
            metadata={
                "links": [GraphStoreLink.incoming(kind="hyperlink", tag="http://b")]
            },
        ),
        Document(id="b", page_content="some text b", metadata={"c": "d"}),
    ]
    assert list(_documents_to_nodes(documents)) == [
        GraphStoreNode(
            id="a",
            metadata={},
            links=[GraphStoreLink.incoming(kind="hyperlink", tag="http://b")],
            text="some text a",
        ),
        GraphStoreNode(id="b", metadata={"c": "d"}, text="some text b"),
    ]
