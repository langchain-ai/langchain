"""Test Graph Database Chain."""

import os

import pytest
from langchain.chains.loading import load_chain
from langchain_openai import ChatOpenAI

from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph


def test_connect_neo4j() -> None:
    """Test that Neo4j database is correctly instantiated and connected."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )

    output = graph.query('RETURN "test" AS output')
    expected_output = [{"output": "test"}]
    assert output == expected_output


def test_connect_neo4j_env() -> None:
    """Test that Neo4j database environment variables."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")
    os.environ["NEO4J_URI"] = url
    os.environ["NEO4J_USERNAME"] = username
    os.environ["NEO4J_PASSWORD"] = password
    graph = Neo4jGraph()

    output = graph.query('RETURN "test" AS output')
    expected_output = [{"output": "test"}]
    assert output == expected_output
    del os.environ["NEO4J_URI"]
    del os.environ["NEO4J_USERNAME"]
    del os.environ["NEO4J_PASSWORD"]


def test_cypher_generating_run() -> None:
    """Test that Cypher statement is correctly generated and executed."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

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
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        allow_dangerous_requests=True,
    )
    output = chain.run("Who acted in Pulp Fiction?")
    assert isinstance(output, str)


def test_cypher_top_k() -> None:
    """Test top_k parameter correctly limits the number of results in the context."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

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
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        return_direct=True,
        top_k=TOP_K,
        allow_dangerous_requests=True,
    )
    output = chain.run("Who acted in Pulp Fiction?")
    assert len(output) == TOP_K


def test_cypher_intermediate_steps() -> None:
    """Test the returning of the intermediate steps."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

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
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )
    output = chain("Who acted in Pulp Fiction?")
    assert isinstance(output, dict)
    assert isinstance(output["result"], str)
    assert "intermediate_steps" in output.keys()
    intermediate_steps = output["intermediate_steps"]
    assert isinstance(intermediate_steps, list)
    assert len(intermediate_steps) == 2
    assert isinstance(intermediate_steps[0], dict)
    assert "query" in intermediate_steps[0].keys()
    query = intermediate_steps[0]["query"]
    assert isinstance(query, str)
    assert isinstance(intermediate_steps[1], dict)
    assert "context" in intermediate_steps[1].keys()
    context = intermediate_steps[1]["context"]
    assert isinstance(context, list)


def test_cypher_return_direct() -> None:
    """Test that chain returns direct results."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

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
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        return_direct=True,
        allow_dangerous_requests=True,
    )
    query = "Who acted in Pulp Fiction?"
    output = chain(query)
    assert isinstance(output, dict)
    assert "query" in output.keys()
    assert query == output["query"]
    assert "result" in output.keys()
    assert isinstance(output["result"], list)


@pytest.mark.skip(reason="load_chain is failing and is due to be deprecated")
def test_cypher_save_load() -> None:
    """Test saving and loading."""

    FILE_PATH = "cypher.yaml"
    url = os.environ.get("NEO4J_URI")
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
    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        return_direct=True,
    )

    chain.save(file_path=FILE_PATH)
    qa_loaded = load_chain(FILE_PATH, graph=graph)

    assert qa_loaded == chain


def test_exclude_types() -> None:
    """Test exclude types from schema."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

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
        "<-[:DIRECTED]-(p:Person {name:'John'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        exclude_types=["Person", "DIRECTED"],
        allow_dangerous_requests=True,
    )
    expected_schema = (
        "Node properties are the following:\n"
        "Actor {name: STRING},Movie {title: STRING}\n"
        "Relationship properties are the following:\n\n"
        "The relationships are the following:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )
    assert chain.graph_schema == expected_schema


def test_include_types() -> None:
    """Test include types from schema."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

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
        "<-[:DIRECTED]-(p:Person {name:'John'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        include_types=["Movie", "Actor", "ACTED_IN"],
        allow_dangerous_requests=True,
    )
    expected_schema = (
        "Node properties are the following:\n"
        "Actor {name: STRING},Movie {title: STRING}\n"
        "Relationship properties are the following:\n\n"
        "The relationships are the following:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )

    assert chain.graph_schema == expected_schema


def test_include_types2() -> None:
    """Test include types from schema."""
    url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

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
        "<-[:DIRECTED]-(p:Person {name:'John'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, seed=0),
        graph=graph,
        include_types=["Movie", "ACTED_IN"],
        allow_dangerous_requests=True,
    )
    expected_schema = (
        "Node properties are the following:\n"
        "Movie {title: STRING}\n"
        "Relationship properties are the following:\n\n"
        "The relationships are the following:\n"
    )
    assert chain.graph_schema == expected_schema
