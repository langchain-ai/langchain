"""Test Graph Database Chain."""
import os

from langchain.chains.graph_qa.aql import ArangoDBGraphQAChain
from langchain.graphs import ArangoDBGraph
from langchain.llms.openai import OpenAI
from arango import ArangoClient
from arango.database import Database

def get_arangodb_client() -> Database:
    url = os.environ.get("ARANGODB_URL", "http://localhost:8529")
    dbname = os.environ.get("ARANGODB_DBNAME", "_system")
    username = os.environ.get("ARANGODB_USERNAME", "root")
    password = os.environ.get("ARANGODB_PASSWORD", "openSesame")

    return ArangoClient(url).db(dbname, username, password, verify=True)

def populate_arangodb_database(db: Database) -> None:
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
        ]
    )

    documents = [
        { "_key": "NedStark", "name": "Ned", "surname": "Stark", "alive": True, "age": 41, "gender": "male"},
        { "_key": "CatelynStark", "name": "Catelyn", "surname": "Stark", "alive": False, "age": 40, "gender": "female"},
        { "_key": "AryaStark", "name": "Arya", "surname": "Stark", "alive": True, "age": 11, "gender": "female"},
        { "_key": "BranStark", "name": "Bran", "surname": "Stark", "alive": True, "age": 10, "gender": "male"},
    ]
    
    edges = [
        {"_to": "Characters/NedStark", "_from": "Characters/AryaStark" },
        {"_to": "Characters/NedStark", "_from": "Characters/BranStark" },
        {"_to": "Characters/CatelynStark", "_from": "Characters/AryaStark" },
        {"_to": "Characters/CatelynStark", "_from": "Characters/BranStark" }
    ]

    db.collection("Characters").import_bulk(documents)
    db.collection("ChildOf").import_bulk(edges)


def empty_arangodb_database(db: Database) -> None:
    for g in db.graphs():
        db.delete_graph(g['name'], drop_collections=True)

    for c in db.collections():
        if not c['system']:
            db.delete_collection(c['name'])


def test_connect_arangodb() -> None:
    """Test that the ArangoDB database is correctly instantiated and connected."""
    graph = ArangoDBGraph(get_arangodb_client())

    sample_aql_result = graph.query("RETURN 'hello world'")
    assert ['hello_world'] == sample_aql_result


def test_aql_generation() -> None:
    """Test that AQL statement is correctly generated and executed."""
    db = get_arangodb_client()
    populate_arangodb_database()

    graph = ArangoDBGraph(db)

    chain = ArangoDBGraphQAChain.from_llm(OpenAI(temperature=0), graph=graph)

    output = chain.run("Is Ned Stark alive?")
    assert output == "Yes, Ned Stark is alive."

    output = chain.run("Is Catelyn Stark alive?")
    assert output == "No, Catelyn Stark is not alive."

    output = chain.run("Are Ned Stark & Catelyn Stark alive?")
    assert output == "Yes, Ned Stark is alive and Catelyn Stark is not alive."

    # chain.run("Does Catelyn Stark have any children?")
    # chain.run("Does Catelyn Stark have any children? Use INBOUND")
    # chain.run("Who is the eldest child of Catelyn Stark? Use INBOUND")
    # chain.run("Do Ned Stark & Catelyn Stark have any children together?")
    # chain.run("Are Arya Stark's parents still alive?")
    # chain.run("Does Bran Stark have a sister?")
    # chain.run("Does Arya Stark have a brother?")