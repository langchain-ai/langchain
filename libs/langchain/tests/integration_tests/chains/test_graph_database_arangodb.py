"""Test Graph Database Chain."""
from typing import Any

from langchain.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.graphs import ArangoGraph
from langchain.graphs.arangodb_graph import get_arangodb_client
from langchain.llms.openai import OpenAI


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


def test_empty_schema_on_no_data() -> None:
    """Test that the schema is empty for an empty ArangoDB Database"""
    db = get_arangodb_client()
    db.delete_graph("GameOfThrones", drop_collections=True, ignore_missing=True)
    db.delete_collection("empty_collection", ignore_missing=True)
    db.create_collection("empty_collection")

    graph = ArangoGraph(db)

    assert graph.schema == {
        "Graph Schema": [],
        "Collection Schema": [],
    }


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
