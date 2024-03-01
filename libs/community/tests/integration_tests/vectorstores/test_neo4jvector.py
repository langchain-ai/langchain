"""Test Neo4jVector functionality."""
import os
from typing import List

from langchain_core.documents import Document

from langchain_community.vectorstores.neo4j_vector import (
    Neo4jVector,
    SearchType,
    _get_search_index_query,
)
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

OS_TOKEN_COUNT = 1536

texts = ["foo", "bar", "baz", "It is the end of the world. Take shelter!"]

"""
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f neo4j.yml up
"""


def drop_vector_indexes(store: Neo4jVector) -> None:
    """Cleanup all vector indexes"""
    all_indexes = store.query(
        """
            SHOW INDEXES YIELD name, type
            WHERE type IN ["VECTOR", "FULLTEXT"]
            RETURN name
                              """
    )
    for index in all_indexes:
        store.query(f"DROP INDEX {index['name']}")


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


def test_neo4jvector() -> None:
    """Test end to end construction and search."""
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_neo4jvector_euclidean() -> None:
    """Test euclidean distance"""
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_neo4jvector_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_neo4jvector_catch_wrong_index_name() -> None:
    """Test if index name is misspelled, but node label and property are correct."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    existing = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="test",
    )
    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(existing)


def test_neo4jvector_catch_wrong_node_label() -> None:
    """Test if node label is misspelled, but index name is correct."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    existing = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="vector",
        node_label="test",
    )
    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(existing)


def test_neo4jvector_with_metadatas() -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    drop_vector_indexes(docsearch)


def test_neo4jvector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]

    drop_vector_indexes(docsearch)


def test_neo4jvector_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9998376369476318),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9993523359298706),
    ]

    drop_vector_indexes(docsearch)


def test_neo4jvector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.9999},
    )
    output = retriever.get_relevant_documents("foo")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
    ]

    drop_vector_indexes(docsearch)


def test_custom_return_neo4jvector() -> None:
    """Test end to end construction and search."""
    docsearch = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        retrieval_query="RETURN 'foo' AS text, score, {test: 'test'} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


def test_neo4jvector_prefer_indexname() -> None:
    """Test using when two indexes are found, prefer by index_name."""
    Neo4jVector.from_texts(
        texts=["foo"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )

    Neo4jVector.from_texts(
        texts=["bar"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        node_label="Test",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    existing_index = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        text_node_property="info",
    )

    output = existing_index.similarity_search("bar", k=1)
    assert output == [Document(page_content="bar", metadata={})]
    drop_vector_indexes(existing_index)


def test_neo4jvector_prefer_indexname_insert() -> None:
    """Test using when two indexes are found, prefer by index_name."""
    Neo4jVector.from_texts(
        texts=["baz"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )

    Neo4jVector.from_texts(
        texts=["foo"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        node_label="Test",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    existing_index = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        text_node_property="info",
    )

    existing_index.add_documents([Document(page_content="bar", metadata={})])

    output = existing_index.similarity_search("bar", k=2)
    assert output == [
        Document(page_content="bar", metadata={}),
        Document(page_content="foo", metadata={}),
    ]
    drop_vector_indexes(existing_index)


def test_neo4jvector_hybrid() -> None:
    """Test end to end construction with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_neo4jvector_hybrid_deduplicate() -> None:
    """Test result deduplication with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search("foo", k=3)
    assert output == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]

    drop_vector_indexes(docsearch)


def test_neo4jvector_hybrid_retrieval_query() -> None:
    """Test custom retrieval_query with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        retrieval_query="RETURN 'moo' AS text, score, {test: 'test'} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="moo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


def test_neo4jvector_hybrid_retrieval_query2() -> None:
    """Test custom retrieval_query with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        retrieval_query="RETURN node.text AS text, score, {test: 'test'} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


def test_neo4jvector_missing_keyword() -> None:
    """Test hybrid search with missing keyword_index_search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    try:
        Neo4jVector.from_existing_index(
            embedding=FakeEmbeddingsWithOsDimension(),
            url=url,
            username=username,
            password=password,
            index_name="vector",
            search_type=SearchType.HYBRID,
        )
    except ValueError as e:
        assert str(e) == (
            "keyword_index name has to be specified when " "using hybrid search option"
        )
    drop_vector_indexes(docsearch)


def test_neo4jvector_hybrid_from_existing() -> None:
    """Test hybrid search with missing keyword_index_search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    existing = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="vector",
        keyword_index_name="keyword",
        search_type=SearchType.HYBRID,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(existing)


def test_neo4jvector_from_existing_graph() -> None:
    """Test from_existing_graph with a single property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'Foo'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name"],
        embedding_node_property="embedding",
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: Foo")]

    drop_vector_indexes(existing)


def test_neo4jvector_from_existing_graph_hybrid() -> None:
    """Test from_existing_graph hybrid with a single property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'foo'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name"],
        embedding_node_property="embedding",
        search_type=SearchType.HYBRID,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: foo")]

    drop_vector_indexes(existing)


def test_neo4jvector_from_existing_graph_multiple_properties() -> None:
    """Test from_existing_graph with a two property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )
    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'Foo', name2: 'Fooz'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)


def test_neo4jvector_from_existing_graph_multiple_properties_hybrid() -> None:
    """Test from_existing_graph with a two property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )
    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'Foo', name2: 'Fooz'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
        search_type=SearchType.HYBRID,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)


def test_neo4jvector_special_character() -> None:
    """Test removing lucene."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search(
        "It is the end of the world. Take shelter!", k=1
    )
    assert output == [
        Document(page_content="It is the end of the world. Take shelter!", metadata={})
    ]

    drop_vector_indexes(docsearch)


def test_hybrid_score_normalization() -> None:
    """Test if we can get two 1.0 documents with RRF"""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(["foo"], text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    # Remove deduplication part of the query
    rrf_query = (
        _get_search_index_query(SearchType.HYBRID)
        .rstrip("WITH node, max(score) AS score ORDER BY score DESC LIMIT $k")
        .replace("UNION", "UNION ALL")
        + "RETURN node.text AS text, score LIMIT 2"
    )

    output = docsearch.query(
        rrf_query,
        params={
            "index": "vector",
            "k": 1,
            "embedding": FakeEmbeddingsWithOsDimension().embed_query("foo"),
            "query": "foo",
            "keyword_index": "keyword",
        },
    )
    # Both FT and Vector must return 1.0 score
    assert output == [{"text": "foo", "score": 1.0}, {"text": "foo", "score": 1.0}]
    drop_vector_indexes(docsearch)


def test_index_fetching() -> None:
    """testing correct index creation and fetching"""
    embeddings = FakeEmbeddings()

    def create_store(
        node_label: str, index: str, text_properties: List[str]
    ) -> Neo4jVector:
        return Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=url,
            username=username,
            password=password,
            index_name=index,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property="embedding",
        )

    def fetch_store(index_name: str) -> Neo4jVector:
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
    create_store("label0", index_0_str, ["text"])

    # create index 1
    index_1_str = "index1"
    create_store("label1", index_1_str, ["text"])

    index_1_store = fetch_store(index_1_str)
    assert index_1_store.index_name == index_1_str

    index_0_store = fetch_store(index_0_str)
    assert index_0_store.index_name == index_0_str
