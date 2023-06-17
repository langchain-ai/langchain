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


def test_cypher_top_k() -> None:
    """Test top_k parameter correctly limits the number of results in the context."""
    url = os.environ.get("NEO4J_URL")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    TOP_K = 1

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
        "<-[:ACTED_IN]-(:Actor {name:'Foo'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(
        OpenAI(temperature=0), graph=graph, return_direct=True, top_k=TOP_K
    )
    output = chain.run("Who played in Pulp Fiction?")
    assert len(output) == TOP_K


def test_cypher_intermediate_steps() -> None:
    """Test the returning of the intermediate steps."""
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

    chain = GraphCypherQAChain.from_llm(
        OpenAI(temperature=0), graph=graph, return_intermediate_steps=True
    )
    output = chain("Who played in Pulp Fiction?")

    expected_output = " Bruce Willis played in Pulp Fiction."
    assert output["result"] == expected_output

    query = output["intermediate_steps"][0]["query"]
    expected_query = (
        "\n\nMATCH (a:Actor)-[:ACTED_IN]->"
        "(m:Movie {title: 'Pulp Fiction'}) RETURN a.name"
    )
    assert query == expected_query

    context = output["intermediate_steps"][1]["context"]
    expected_context = [{"a.name": "Bruce Willis"}]
    assert context == expected_context


def test_cypher_return_direct() -> None:
    """Test that chain returns direct results."""
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

    chain = GraphCypherQAChain.from_llm(
        OpenAI(temperature=0), graph=graph, return_direct=True
    )
    output = chain.run("Who played in Pulp Fiction?")
    expected_output = [{"a.name": "Bruce Willis"}]
    assert output == expected_output
