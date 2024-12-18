"""
Integration tests for FalkorDB vector store functionality.

These tests validate the end-to-end process of constructing, indexing,
and searching vector embeddings in a FalkorDB instance. They include:
- Setting up the FalkorDB vector store with a local instance.
- Indexing documents with fake embeddings.
- Performing vector searches and validating results.

Note:
These tests are conducted using a local FalkorDB instance but can also
be run against a Cloud FalkorDB instance. Ensure that appropriate host
and port configurations are set up before running the tests.
"""

import os
from math import isclose
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_community.vectorstores.falkordb_vector import (
    FalkorDBVector,
    SearchType,
    process_index_data,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

# Load environment variables from .env file
load_dotenv()

host = os.getenv("FALKORDB_HOST", "localhost")
port = int(os.getenv("FALKORDB_PORT", 6379))

OS_TOKEN_COUNT = 1535

texts = ["foo", "bar", "baz", "It is the end of the world. Take shelter!"]


def drop_vector_indexes(store: FalkorDBVector) -> None:
    """Cleanup all vector indexes"""
    index_entity_labels: List[Any] = []
    index_entity_properties: List[Any] = []
    index_entity_types: List[Any] = []

    # get all indexes
    result = store._query(
        """
    CALL db.indexes()
    """
    )
    processed_result: List[Dict[str, Any]] = process_index_data(result)

    # get all vector indexs entity labels, entity properties, entity_types
    if isinstance(processed_result, list):
        for index in processed_result:
            if isinstance(index, dict):
                if index.get("index_type") == "VECTOR":
                    index_entity_labels.append(index["entity_label"])
                    index_entity_properties.append(index["entity_property"])
                    index_entity_types.append(index["entity_type"])

    # drop vector indexs
    for entity_label, entity_property, entity_type in zip(
        index_entity_labels, index_entity_properties, index_entity_types
    ):
        if entity_type == "NODE":
            store._database.drop_node_vector_index(
                label=entity_label,
                attribute=entity_property,
            )
        elif entity_type == "RELATIONSHIP":
            store._database.drop_edge_vector_index(
                label=entity_label,
                attribute=entity_property,
            )


class FakeEmbeddingsWithOsDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(i + 1)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(texts.index(text) + 1)]


def test_falkordbvector() -> None:
    """Test end to end construction and search."""
    docsearch = FalkorDBVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert type(output) is list
    assert type(output[0]) is Document
    assert output[0].page_content == "foo"

    drop_vector_indexes(docsearch)


def test_falkordbvector_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert type(output) is list
    assert type(output[0]) is Document
    assert output[0].page_content == "foo"

    drop_vector_indexes(docsearch)


def test_falkordbvector_catch_wrong_node_label() -> None:
    """Test if node label is misspelled, but index name is correct."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    try:
        FalkorDBVector.from_existing_index(
            embedding=FakeEmbeddingsWithOsDimension(),
            host=host,
            port=port,
            node_label="test",
        )
    except Exception as e:
        assert type(e) is ValueError
        assert str(e) == (
            "The specified vector index node label "
            + "`test` does not exist. Make sure to"
            + " check if you spelled the node label correctly"
        )
        drop_vector_indexes(docsearch)


def test_falkordbvector_with_metadatas() -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = FalkorDBVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert type(output) is list
    assert type(output[0]) is Document
    assert output[0].metadata.get("page") == "0"

    drop_vector_indexes(docsearch)


def test_falkordbvector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = FalkorDBVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    output = [
        (doc, round(score, 1))
        for doc, score in docsearch.similarity_search_with_score("foo", k=1)
    ]
    assert output == [
        (
            Document(
                metadata={
                    "text": "foo",
                    "id": "acbd18db4cc2f85cedef654fccc4a4d8",
                    "page": "0",
                },
                page_content="foo",
            ),
            0.0,
        )
    ]
    drop_vector_indexes(docsearch)


def test_falkordb_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-2."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = FalkorDBVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    expected_output = [
        (
            Document(
                metadata={
                    "text": "foo",
                    "id": "acbd18db4cc2f85cedef654fccc4a4d8",
                    "page": "0",
                },
                page_content="foo",
            ),
            0.0,
        ),
        (
            Document(
                metadata={
                    "text": "bar",
                    "id": "37b51d194a7513e45b56f6524f2d51f2",
                    "page": "1",
                },
                page_content="bar",
            ),
            1.0,
        ),
        (
            Document(
                metadata={
                    "text": "baz",
                    "id": "73feffa4b7f6bb68e44cf984c85f6e88",
                    "page": "2",
                },
                page_content="baz",
            ),
            2.0,
        ),
    ]

    # Check if the length of the outputs matches
    assert len(output) == len(expected_output)

    # Check if each document and its relevance score is close to the expected value
    for (doc, score), (expected_doc, expected_score) in zip(output, expected_output):
        assert doc.page_content == expected_doc.page_content
        assert doc.metadata == expected_doc.metadata
        assert isclose(score, expected_score, rel_tol=1e-5)

    drop_vector_indexes(docsearch)


def test_falkordbvector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = FalkorDBVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.9999},
    )
    output = retriever.invoke("foo")
    assert output == [
        Document(
            metadata={
                "text": "foo",
                "id": "acbd18db4cc2f85cedef654fccc4a4d8",
                "page": "0",
            },
            page_content="foo",
        )
    ]

    drop_vector_indexes(docsearch)


def test_custom_return_falkordbvector() -> None:
    """Test end to end construction and search."""
    docsearch = FalkorDBVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
        retrieval_query="RETURN 'foo' AS text, score, {test: 'test'} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


def test_falkordb_hybrid() -> None:
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [
        Document(
            metadata={"text": "foo", "id": "acbd18db4cc2f85cedef654fccc4a4d8"},
            page_content="foo",
        )
    ]

    drop_vector_indexes(docsearch)


def test_falkordb_hybrid_deduplicate() -> None:
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search("foo", k=3)
    assert output == [
        Document(
            metadata={"text": "baz", "id": "73feffa4b7f6bb68e44cf984c85f6e88"},
            page_content="baz",
        ),
        Document(
            metadata={"text": "foo", "id": "acbd18db4cc2f85cedef654fccc4a4d8"},
            page_content="foo",
        ),
        Document(
            metadata={"text": "bar", "id": "37b51d194a7513e45b56f6524f2d51f2"},
            page_content="bar",
        ),
    ]

    drop_vector_indexes(docsearch)


def test_falkordb_hybrid_retrieval_query() -> None:
    """Test custom retrieval_query with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        retrieval_query="RETURN 'moo' AS text, score, {test: 'test'} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="moo", metadata={"test": "test"})]
    drop_vector_indexes(docsearch)


def test_falkordbvector_missing_keyword() -> None:
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    node_label = "vector"
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    try:
        FalkorDBVector.from_existing_index(
            embedding=FakeEmbeddingsWithOsDimension(),
            host=host,
            port=port,
            node_label=node_label,
            search_type=SearchType.HYBRID,
        )
    except Exception as e:
        assert str(e) == (
            "The specified vector index node label "
            + f"`{node_label}` does not exist. Make sure"
            + " to check if you spelled the node label correctly"
        )

    drop_vector_indexes(docsearch)


def test_falkordb_hybrid_from_existing() -> None:
    """Test hybrid search with missing keyword_index_search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    existing = FalkorDBVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        node_label="Chunk",  # default node label
        search_type=SearchType.HYBRID,
    )
    output = existing.similarity_search("foo", k=1)
    assert output == [
        Document(
            metadata={"text": "foo", "id": "acbd18db4cc2f85cedef654fccc4a4d8"},
            page_content="foo",
        )
    ]

    drop_vector_indexes(existing)
    drop_vector_indexes(docsearch)


def test_falkordbvector_from_existing_graph() -> None:
    """Test from_existing_graph with a single property"""
    graph = FalkorDBVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )
    graph._query("MATCH (n) DELETE n")
    graph._query("CREATE (:Test {name:'Foo'}), (:Test {name:'Bar'})")
    assert graph.database_name, "Database name cannot be empty or None"
    existing = FalkorDBVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        database=graph.database_name,
        host=host,
        port=port,
        node_label="Test",
        text_node_properties=["name"],
        embedding_node_property="embedding",
    )

    output = existing.similarity_search("foo", k=2)

    assert [output[0]] == [Document(page_content="\nname: Foo")]

    drop_vector_indexes(existing)


def test_falkordb_from_existing_graph_mulitiple_properties() -> None:
    """Test from_existing_graph with two properties."""
    graph = FalkorDBVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )
    graph._query("MATCH (n) DELETE n")
    graph._query("CREATE (:Test {name:'Foo', name2: 'Fooz'}), (:Test {name:'Bar'})")
    assert graph.database_name, "Database name cannot be empty or None"
    existing = FalkorDBVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        database=graph.database_name,
        host=host,
        port=port,
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
    )

    output = existing.similarity_search("foo", k=2)
    assert [output[0]] == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)
    drop_vector_indexes(graph)


def test_falkordbvector_special_character() -> None:
    """Test removing lucene."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = FalkorDBVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search(
        "It is the end of the world. Take shelter!", k=1
    )

    assert output == [
        Document(
            metadata={
                "text": "It is the end of the world. Take shelter!",
                "id": "84768c9c477cbe05fbafbe7247990051",
            },
            page_content="It is the end of the world. Take shelter!",
        )
    ]
    drop_vector_indexes(docsearch)


def test_falkordb_from_existing_graph_mulitiple_properties_hybrid() -> None:
    """Test from_existing_graph with a two property."""
    graph = FalkorDBVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        host=host,
        port=port,
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    graph._query("MATCH (n) DELETE n")
    graph._query("CREATE (:Test {name:'Foo', name2: 'Fooz'}), (:Test {name:'Bar'})")
    assert graph.database_name, "Database name cannot be empty or None"
    existing = FalkorDBVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        database=graph.database_name,
        host=host,
        port=port,
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
        search_type=SearchType.HYBRID,
    )

    output = existing.similarity_search("foo", k=2)

    assert [output[0]] == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)


def test_index_fetching() -> None:
    """testing correct index creation and fetching"""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    embeddings = FakeEmbeddingsWithOsDimension()

    def create_store(node_label: str, text_properties: List[str]) -> FalkorDBVector:
        return FalkorDBVector.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=FakeEmbeddingsWithOsDimension(),
            node_label=node_label,
            host=host,
            port=port,
            pre_delete_collection=True,
        )

    def fetch_store(node_label: str) -> FalkorDBVector:
        store = FalkorDBVector.from_existing_index(
            embedding=embeddings,
            host=host,
            port=port,
            node_label=node_label,
        )
        return store

    index_0_str = "label0"
    create_store(index_0_str, ["text"])

    # create index 1
    index_1_str = "label1"
    create_store("label1", ["text"])

    index_1_store = fetch_store(index_1_str)
    assert index_1_store.node_label == index_1_str

    index_0_store = fetch_store(index_0_str)
    assert index_0_store.node_label == index_0_str

    drop_vector_indexes(index_1_store)
    drop_vector_indexes(index_0_store)


def test_retrieval_params() -> None:
    """Test if we use parameters in retrieval query"""
    docsearch = FalkorDBVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        retrieval_query="""
        RETURN $test as text, score, {test: $test1} AS metadata
        """,
    )
    output = docsearch.similarity_search(
        "Foo", k=2, params={"test": "test", "test1": "test1"}
    )
    assert output == [
        Document(page_content="test", metadata={"test": "test1"}),
        Document(page_content="test", metadata={"test": "test1"}),
    ]
    drop_vector_indexes(docsearch)


def test_falkordb_relationship_index() -> None:
    """Test end to end construction and search."""
    embeddings = FakeEmbeddingsWithOsDimension()
    docsearch = FalkorDBVector.from_texts(
        texts=texts,
        embedding=embeddings,
        host=host,
        port=port,
        pre_delete_collection=True,
    )
    # Ingest data
    docsearch._query(
        (
            "MERGE (p1:Person)"
            "MERGE (p2:Person)"
            "MERGE (p3:Person)"
            "MERGE (p4:Person)"
            "MERGE (p1)-[:REL {text: 'foo', embedding: vecf32($e1)}]->(p2)"
            "MERGE (p3)-[:REL {text: 'far', embedding: vecf32($e2)}]->(p4)"
        ),
        params={
            "e1": embeddings.embed_query("foo"),
            "e2": embeddings.embed_query("bar"),
        },
    )
    # Create relationship index
    docsearch.create_new_index_on_relationship(
        relation_type="REL",
        embedding_node_property="embedding",
        embedding_dimension=OS_TOKEN_COUNT,
    )
    relationship_index = FalkorDBVector.from_existing_relationship_index(
        embeddings, relation_type="REL"
    )
    output = relationship_index.similarity_search("foo", k=1)
    assert output == [Document(metadata={"text": "foo"}, page_content="foo")]

    drop_vector_indexes(docsearch)
    drop_vector_indexes(relationship_index)
