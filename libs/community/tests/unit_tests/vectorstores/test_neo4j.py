"""Test Neo4j functionality."""
import os
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from libs.community.langchain_community.vectorstores.neo4j_vector import Neo4jVector


def test_escaping_lucene() -> None:
    """Test escaping lucene characters"""
    assert remove_lucene_chars("Hello+World") == "Hello World"
    assert remove_lucene_chars("Hello World\\") == "Hello World"
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter!")
            == "It is the end of the world. Take shelter"
    )
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter&&")
            == "It is the end of the world. Take shelter"
    )
    assert (
            remove_lucene_chars("Bill&&Melinda Gates Foundation")
            == "Bill  Melinda Gates Foundation"
    )
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter(&&)")
            == "It is the end of the world. Take shelter"
    )
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter??")
            == "It is the end of the world. Take shelter"
    )
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter^")
            == "It is the end of the world. Take shelter"
    )
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter+")
            == "It is the end of the world. Take shelter"
    )
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter-")
            == "It is the end of the world. Take shelter"
    )
    assert (
            remove_lucene_chars("It is the end of the world. Take shelter~")
            == "It is the end of the world. Take shelter"
    )


def test_index_fetching() -> None:
    """testing correct index creation and fetching"""
    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    embeddings = FakeEmbeddings()

    def create_store(node_label, index, text_properties) -> Neo4jVector:
        return Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=url,
            username=username,
            password=password,
            index_name=index,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property=f"embedding",
        )

    def fetch_store(index_name) -> Neo4jVector:
        store = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=url,
            username=username,
            password=password,
            index_name=index_name,
        )
        return store

    # create index 0
    index_0_str = "index0"
    create_store('label0', index_0_str, ['text'])

    # create index 1
    index_1_str = "index1"
    create_store('label1', index_1_str, ['text'])

    index_1_store = fetch_store(index_1_str)
    assert index_1_store.index_name == index_1_str

    index_0_store = fetch_store(index_0_str)
    assert index_0_store.index_name == index_0_str


