"""Test Neo4jVector functionality."""

import os
from math import isclose
from typing import Any, Dict, List, cast

from langchain_core.documents import Document
from yaml import safe_load

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import (
    Neo4jVector,
    SearchType,
    _get_search_index_query,
)
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    FakeEmbeddings,
)
from tests.integration_tests.vectorstores.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
)

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
        store.query(f"DROP INDEX `{index['name']}`")

    store.query("MATCH (n) DETACH DELETE n;")


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
    output = [
        (doc, round(score, 1))
        for doc, score in docsearch.similarity_search_with_score("foo", k=1)
    ]
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
    expected_output = [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9998376369476318),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9993523359298706),
    ]

    # Check if the length of the outputs matches
    assert len(output) == len(expected_output)

    # Check if each document and its relevance score is close to the expected value
    for (doc, score), (expected_doc, expected_score) in zip(output, expected_output):
        assert doc.page_content == expected_doc.page_content
        assert doc.metadata == expected_doc.metadata
        assert isclose(score, expected_score, rel_tol=1e-5)

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
    output = retriever.invoke("foo")
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
            "keyword_index name has to be specified when using hybrid search option"
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

    graph.query("CREATE (:Test {name:'Foo'}),(:Test {name:'Bar'})")

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

    graph.query("CREATE (:Test {name:'foo'}),(:Test {name:'Bar'})")

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

    graph.query("CREATE (:Test {name:'Foo', name2: 'Fooz'}),(:Test {name:'Bar'})")

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

    graph.query("CREATE (:Test {name:'Foo', name2: 'Fooz'}),(:Test {name:'Bar'})")

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
    drop_vector_indexes(index_1_store)
    drop_vector_indexes(index_0_store)


def test_retrieval_params() -> None:
    """Test if we use parameters in retrieval query"""
    docsearch = Neo4jVector.from_texts(
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


def test_retrieval_dictionary() -> None:
    """Test if we use parameters in retrieval query"""
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        retrieval_query="""
        RETURN {
            name:'John', 
            age: 30,
            skills: ["Python", "Data Analysis", "Machine Learning"]} as text, 
            score, {} AS metadata
        """,
    )
    expected_output = [
        Document(
            page_content=(
                "skills:\n- Python\n- Data Analysis\n- "
                "Machine Learning\nage: 30\nname: John\n"
            )
        )
    ]

    output = docsearch.similarity_search("Foo", k=1)

    def parse_document(doc: Document) -> Any:
        return safe_load(doc.page_content)

    parsed_expected = [parse_document(doc) for doc in expected_output]
    parsed_output = [parse_document(doc) for doc in output]

    assert parsed_output == parsed_expected
    drop_vector_indexes(docsearch)


def test_metadata_filters_type1() -> None:
    """Test metadata filters"""
    docsearch = Neo4jVector.from_documents(
        DOCUMENTS,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
    )
    # We don't test type 5, because LIKE has very SQL specific examples
    for example in (
        TYPE_1_FILTERING_TEST_CASES
        + TYPE_2_FILTERING_TEST_CASES
        + TYPE_3_FILTERING_TEST_CASES
        + TYPE_4_FILTERING_TEST_CASES
    ):
        filter_dict = cast(Dict[str, Any], example[0])
        output = docsearch.similarity_search("Foo", filter=filter_dict)
        indices = cast(List[int], example[1])
        adjusted_indices = [index - 1 for index in indices]
        expected_output = [DOCUMENTS[index] for index in adjusted_indices]
        # We don't return id properties from similarity search by default
        # Also remove any key where the value is None
        for doc in expected_output:
            if "id" in doc.metadata:
                del doc.metadata["id"]
            keys_with_none = [
                key for key, value in doc.metadata.items() if value is None
            ]
            for key in keys_with_none:
                del doc.metadata[key]

        assert output == expected_output
    drop_vector_indexes(docsearch)


def test_neo4jvector_relationship_index() -> None:
    """Test end to end construction and search."""
    embeddings = FakeEmbeddingsWithOsDimension()
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    # Ingest data
    docsearch.query(
        (
            "CREATE ()-[:REL {text: 'foo', embedding: $e1}]->()"
            ", ()-[:REL {text: 'far', embedding: $e2}]->()"
        ),
        params={
            "e1": embeddings.embed_query("foo"),
            "e2": embeddings.embed_query("bar"),
        },
    )
    # Create relationship index
    docsearch.query(
        """CREATE VECTOR INDEX `relationship`
FOR ()-[r:REL]-() ON (r.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
"""
    )
    relationship_index = Neo4jVector.from_existing_relationship_index(
        embeddings, index_name="relationship"
    )

    output = relationship_index.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_neo4jvector_relationship_index_retrieval() -> None:
    """Test end to end construction and search."""
    embeddings = FakeEmbeddingsWithOsDimension()
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        pre_delete_collection=True,
    )
    # Ingest data
    docsearch.query(
        (
            "CREATE ({node:'text'})-[:REL {text: 'foo', embedding: $e1}]->()"
            ", ({node:'text'})-[:REL {text: 'far', embedding: $e2}]->()"
        ),
        params={
            "e1": embeddings.embed_query("foo"),
            "e2": embeddings.embed_query("bar"),
        },
    )
    # Create relationship index
    docsearch.query(
        """CREATE VECTOR INDEX `relationship`
FOR ()-[r:REL]-() ON (r.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
"""
    )
    retrieval_query = (
        "RETURN relationship.text + '-' + startNode(relationship).node "
        "AS text, score, {foo:'bar'} AS metadata"
    )
    relationship_index = Neo4jVector.from_existing_relationship_index(
        embeddings, index_name="relationship", retrieval_query=retrieval_query
    )

    output = relationship_index.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo-text", metadata={"foo": "bar"})]

    drop_vector_indexes(docsearch)


def test_neo4j_max_marginal_relevance_search() -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==3 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order).
    """
    texts = ["-0.124", "+0.127", "+0.25", "+1.0"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts,
        metadatas=metadatas,
        embedding=AngularTwoDimensionalEmbeddings(),
        pre_delete_collection=True,
    )

    expected_set = {
        ("+0.25", 2),
        ("-0.124", 0),
    }

    output = docsearch.max_marginal_relevance_search("0.0", k=2, fetch_k=3)
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == expected_set

    drop_vector_indexes(docsearch)


def test_neo4jvector_passing_graph_object() -> None:
    """Test end to end construction and search with passing graph object."""
    graph = Neo4jGraph()
    # Rewrite env vars to make sure it fails if env is used
    os.environ["NEO4J_URI"] = "foo"
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        graph=graph,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)
