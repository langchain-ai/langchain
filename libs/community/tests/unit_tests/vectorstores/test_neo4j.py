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

    def create_index(node_label, index, text_properties):
        return Neo4jVector.from_existing_graph(
            embedding=FakeEmbeddings(),
            url=url,
            username=username,
            password=password,
            index_name=index,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property=f"embedding",
        )

    # create index 0
    store_0 = create_index('label0', 'index0', ['text'])
    assert store_0.index_name == "index0"
    # create index 1
    store_1 = create_index('label1', 'index1', ['text'])
    assert store_1.index_name == "index1"
    # create index 2
    store_2 = create_index('label2', 'index2', ["text"])
    assert store_2.index_name == "index2"
    assert store_2.index_name != store_2.index_name

