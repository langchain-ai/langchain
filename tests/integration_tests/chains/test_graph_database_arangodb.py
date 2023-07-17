"""Test Graph Database Chain."""
import os
from typing import Any

from langchain.chains.graph_qa.aql import ArangoDBGraphQAChain
from langchain.graphs import ArangoDBGraph
from langchain.llms.openai import OpenAI


def get_arangodb_client() -> Any:
    from arango import ArangoClient

    url = os.environ.get("ARANGODB_URL", "http://localhost:8529")
    dbname = os.environ.get("ARANGODB_DBNAME", "_system")
    username = os.environ.get("ARANGODB_USERNAME", "root")
    password = os.environ.get("ARANGODB_PASSWORD", "openSesame")

    return ArangoClient(url).db(dbname, username, password, verify=True)


def populate_arangodb_database(db: Any) -> None:
    if db.has_graph("GameOfThrones"):
        return

    db.create_graph(
        "GameOfThrones",
        edge_definitions=[
            {
                "edge_collection": "ChildOf",
                "from_vertex_collections": ["Characters"],
                "to_vertex_collections": ["Characters"],
            },
        ],
    )

    documents = [
        {
            "_key": "NedStark",
            "name": "Ned",
            "surname": "Stark",
            "alive": True,
            "age": 41,
            "gender": "male",
        },
        {
            "_key": "AryaStark",
            "name": "Arya",
            "surname": "Stark",
            "alive": True,
            "age": 11,
            "gender": "female",
        },
    ]

    edges = [{"_to": "Characters/NedStark", "_from": "Characters/AryaStark"}]

    db.collection("Characters").import_bulk(documents)
    db.collection("ChildOf").import_bulk(edges)


def test_connect_arangodb() -> None:
    """Test that the ArangoDB database is correctly instantiated and connected."""
    graph = ArangoDBGraph(get_arangodb_client())

    sample_aql_result = graph.query("RETURN 'hello world'")
    assert ["hello_world"] == sample_aql_result


def test_aql_generation() -> None:
    """Test that AQL statement is correctly generated and executed."""
    db = get_arangodb_client()

    populate_arangodb_database(db)

    graph = ArangoDBGraph(db)
    chain = ArangoDBGraphQAChain.from_llm(OpenAI(temperature=0), graph=graph)

    output = chain.run("Is Ned Stark alive?")
    assert output == "Yes, Ned Stark is alive."

    output = chain.run("How old is Arya Stark?")
    assert output == "Arya Stark is 11 years old."

    output = chain.run("Who is the child of Ned Stark?")
    assert output == "The child of Ned Stark is Arya Stark."
