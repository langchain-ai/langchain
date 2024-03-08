import os

from langchain_community.graphs import MemgraphGraph


def test_cypher_return_correct_schema() -> None:
    """Test that chain returns direct results."""
    url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    username = os.environ.get("MEMGRAPH_USERNAME", "")
    password = os.environ.get("MEMGRAPH_PASSWORD", "")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = MemgraphGraph(
        url=url,
        username=username,
        password=password,
    )
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
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
    relationships = graph.query(
        "CALL llm_util.schema('raw') YIELD schema "
        "WITH schema.relationships AS relationships "
        "UNWIND relationships AS relationship "
        "RETURN relationship['start'] AS start, "
        "relationship['type'] AS type, "
        "relationship['end'] AS end "
        "ORDER BY start, type, end;"
    )

    node_props = graph.query(
        "CALL llm_util.schema('raw') YIELD schema "
        "WITH schema.node_props AS nodes "
        "WITH nodes['LabelA'] AS properties "
        "UNWIND properties AS property "
        "RETURN property['property'] AS prop, "
        "property['type'] AS type "
        "ORDER BY prop ASC;"
    )

    expected_relationships = [
        {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"},
        {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"},
    ]

    expected_node_props = [{"prop": "property_a", "type": "str"}]

    assert relationships == expected_relationships
    assert node_props == expected_node_props
