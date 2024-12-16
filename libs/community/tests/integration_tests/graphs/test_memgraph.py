import os

from langchain_core.documents import Document

from langchain_community.graphs import MemgraphGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.memgraph_graph import NODE_PROPERTIES_QUERY, REL_QUERY

test_data = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo"), Node(id="bar", type="bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="foo"),
                target=Node(id="bar", type="bar"),
                type="REL",
            )
        ],
        source=Document(page_content="source document"),
    )
]


def test_cypher_return_correct_schema() -> None:
    """Test that chain returns direct results."""

    url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    username = os.environ.get("MEMGRAPH_USERNAME", "")
    password = os.environ.get("MEMGRAPH_PASSWORD", "")

    assert url is not None
    assert username is not None
    assert password is not None

    graph = MemgraphGraph(url=url, username=username, password=password)

    # Drop graph
    graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    graph.query("DROP GRAPH")
    graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

    # Create two nodes and a relationship
    graph.query(
        """
        CREATE (la:LabelA {property_a: 'a'})
        CREATE (lb:LabelB)
        CREATE (lc:LabelC)
        MERGE (la)-[:REL_TYPE]-> (lb)
        MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
        """
    )
    # Refresh schema information
    graph.refresh_schema()

    node_properties = graph.query(NODE_PROPERTIES_QUERY)
    relationships = graph.query(REL_QUERY)

    expected_node_properties = [
        {
            "output": {
                "labels": ":`LabelA`",
                "properties": [{"key": "property_a", "types": ["String"]}],
            }
        },
        {"output": {"labels": ":`LabelB`", "properties": [{"key": "", "types": []}]}},
        {"output": {"labels": ":`LabelC`", "properties": [{"key": "", "types": []}]}},
    ]

    expected_relationships = [
        {
            "start_node_labels": ["LabelA"],
            "rel_type": "REL_TYPE",
            "end_node_labels": ["LabelC"],
            "properties_info": [["rel_prop", "STRING"]],
        },
        {
            "start_node_labels": ["LabelA"],
            "rel_type": "REL_TYPE",
            "end_node_labels": ["LabelB"],
            "properties_info": [],
        },
    ]

    graph.close()

    assert node_properties == expected_node_properties
    assert relationships == expected_relationships


def test_add_graph_documents() -> None:
    """Test that Memgraph correctly imports graph document."""
    url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    username = os.environ.get("MEMGRAPH_USERNAME", "")
    password = os.environ.get("MEMGRAPH_PASSWORD", "")

    assert url is not None
    assert username is not None
    assert password is not None

    graph = MemgraphGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
    # Drop graph
    graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    graph.query("DROP GRAPH")
    graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")
    # Create KG
    graph.add_graph_documents(test_data)
    output = graph.query("MATCH (n) RETURN labels(n) AS label, count(*) AS count")
    # Close the connection
    graph.close()
    assert output == [{"label": ["bar"], "count": 1}, {"label": ["foo"], "count": 1}]


def test_add_graph_documents_base_entity() -> None:
    """Test that Memgraph correctly imports graph document with Entity label."""
    url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    username = os.environ.get("MEMGRAPH_USERNAME", "")
    password = os.environ.get("MEMGRAPH_PASSWORD", "")

    assert url is not None
    assert username is not None
    assert password is not None

    graph = MemgraphGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
    # Drop graph
    graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    graph.query("DROP GRAPH")
    graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")
    # Create KG
    graph.add_graph_documents(test_data, baseEntityLabel=True)
    output = graph.query("MATCH (n) RETURN labels(n) AS label, count(*) AS count")

    # Close the connection
    graph.close()

    assert output == [
        {"label": ["__Entity__", "bar"], "count": 1},
        {"label": ["__Entity__", "foo"], "count": 1},
    ]


def test_add_graph_documents_include_source() -> None:
    """Test that Memgraph correctly imports graph document with source included."""
    url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    username = os.environ.get("MEMGRAPH_USERNAME", "")
    password = os.environ.get("MEMGRAPH_PASSWORD", "")

    assert url is not None
    assert username is not None
    assert password is not None

    graph = MemgraphGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
    # Drop graph
    graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    graph.query("DROP GRAPH")
    graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")
    # Create KG
    graph.add_graph_documents(test_data, include_source=True)
    output = graph.query("MATCH (n) RETURN labels(n) AS label, count(*) AS count")

    # Close the connection
    graph.close()

    assert output == [
        {"label": ["bar"], "count": 1},
        {"label": ["foo"], "count": 1},
        {"label": ["Document"], "count": 1},
    ]
