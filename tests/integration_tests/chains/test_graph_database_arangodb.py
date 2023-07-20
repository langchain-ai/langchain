"""Test Graph Database Chain."""
import os
from typing import Any

from langchain.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.graphs import ArangoGraph
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
    graph = ArangoGraph(get_arangodb_client())

    sample_aql_result = graph.query("RETURN 'hello world'")
    assert ["hello_world"] == sample_aql_result


def test_aql_generation() -> None:
    """Test that AQL statement is correctly generated and executed."""
    db = get_arangodb_client()

    populate_arangodb_database(db)

    graph = ArangoGraph(db)
    chain = ArangoGraphQAChain.from_llm(OpenAI(temperature=0), graph=graph)
    chain.return_aql_result = True

    output = chain("Is Ned Stark alive?")
    assert output["aql_result"] == [True]
    assert "Yes" in output["result"]

    output = chain("How old is Arya Stark?")
    assert output["aql_result"] == [11]
    assert "11" in output["result"]

    output = chain("What is the relationship between Arya Stark and Ned Stark?")
    assert len(output["aql_result"]) == 1
    assert "child of" in output["result"]
