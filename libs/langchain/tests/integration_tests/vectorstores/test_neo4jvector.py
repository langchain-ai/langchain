"""Test Neo4jVector functionality."""
import os
from typing import List

from langchain.docstore.document import Document
from langchain.vectorstores import Neo4jVector
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")

OS_TOKEN_COUNT = 1536

texts = ["foo", "bar", "baz"]

"""
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f neo4j.yml up
"""


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


def test_neo4jvector_catch_wrong_index_name() -> None:
    """Test if index name is misspelled, but node label and property are correct."""
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
    existing = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        username=username,
        password=password,
        index_name="test",
    )
    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_neo4jvector_catch_wrong_node_label() -> None:
    """Test if node label is misspelled, but index name is correct."""
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
    print(output)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9998376369476318),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9993523359298706),
    ]


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
