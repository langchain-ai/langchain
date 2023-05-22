"""Test Graph Database Chain."""
import os

from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.llms.openai import OpenAI


def test_connect_neo4j() -> None:
    """Test that Neo4j database is correctly instantiated and connected."""
    url = os.environ.get("NEO4J_URL")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )

    output = graph.query(
        """
    RETURN "test" AS output
    """
    )
    expected_output = [{"output": "test"}]
    assert output == expected_output


def test_cypher_generating_run() -> None:
    """Test that Cypher statement is correctly generated and executed."""
    url = os.environ.get("NEO4J_URL")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(OpenAI(temperature=0), graph=graph)
    output = chain.run("Who played in Pulp Fiction?")
    expected_output = " Bruce Willis played in Pulp Fiction."
    assert output == expected_output
