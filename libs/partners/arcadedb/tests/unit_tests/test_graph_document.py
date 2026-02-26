from langchain_arcadedb import GraphDocument, Node, Relationship


def test_node_defaults() -> None:
    n = Node(id="1")
    assert n.type == "Node"
    assert n.properties == {}


def test_relationship() -> None:
    a = Node(id="1", type="Person")
    b = Node(id="2", type="Person")
    r = Relationship(source=a, target=b, type="KNOWS")
    assert r.type == "KNOWS"
    assert r.source.id == "1"
    assert r.target.id == "2"
    assert r.properties == {}


def test_graph_document() -> None:
    doc = GraphDocument(
        nodes=[Node(id="1", type="Person")],
        relationships=[],
    )
    assert len(doc.nodes) == 1
    assert len(doc.relationships) == 0
    assert doc.source is None
