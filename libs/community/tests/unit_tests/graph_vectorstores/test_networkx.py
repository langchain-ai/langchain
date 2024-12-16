import pytest
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.links import METADATA_LINKS_KEY, Link
from langchain_community.graph_vectorstores.networkx import documents_to_networkx


@pytest.mark.requires("networkx")
def test_documents_to_networkx() -> None:
    import networkx as nx

    doc1 = Document(
        id="a",
        page_content="some content",
        metadata={
            METADATA_LINKS_KEY: [
                Link.incoming("href", "a"),
                Link.bidir("kw", "foo"),
            ]
        },
    )
    doc2 = Document(
        id="b",
        page_content="<some\n more content>",
        metadata={
            METADATA_LINKS_KEY: [
                Link.incoming("href", "b"),
                Link.outgoing("href", "a"),
                Link.bidir("kw", "foo"),
                Link.bidir("kw", "bar"),
            ]
        },
    )

    graph_with_tags = documents_to_networkx([doc1, doc2], tag_nodes=True)
    link_data = nx.node_link_data(graph_with_tags)
    assert link_data["directed"]
    assert not link_data["multigraph"]

    link_data["nodes"].sort(key=lambda n: n["id"])
    assert link_data["nodes"] == [
        {"id": "a", "text": "some content"},
        {"id": "b", "text": "<some\n more content>"},
        {"id": "tag_0", "label": "href:a"},
        {"id": "tag_1", "label": "kw:foo"},
        {"id": "tag_2", "label": "href:b"},
        {"id": "tag_3", "label": "kw:bar"},
    ]
    link_data["links"].sort(key=lambda n: (n["source"], n["target"]))
    assert link_data["links"] == [
        {"source": "a", "target": "tag_1"},
        {"source": "b", "target": "tag_0"},
        {"source": "b", "target": "tag_1"},
        {"source": "b", "target": "tag_3"},
        {"source": "tag_0", "target": "a"},
        {"source": "tag_1", "target": "a"},
        {"source": "tag_1", "target": "b"},
        {"source": "tag_2", "target": "b"},
        {"source": "tag_3", "target": "b"},
    ]

    graph_without_tags = documents_to_networkx([doc1, doc2], tag_nodes=False)
    link_data = nx.node_link_data(graph_without_tags)
    assert link_data["directed"]
    assert not link_data["multigraph"]

    link_data["nodes"].sort(key=lambda n: n["id"])
    assert link_data["nodes"] == [
        {"id": "a", "text": "some content"},
        {"id": "b", "text": "<some\n more content>"},
    ]

    link_data["links"].sort(key=lambda n: (n["source"], n["target"]))
    assert link_data["links"] == [
        {"source": "a", "target": "b", "label": "['kw:foo']"},
        {"source": "b", "target": "a", "label": "['href:a', 'kw:foo']"},
    ]
