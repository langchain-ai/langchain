"""Test TiDB Vector functionality."""
import os
from typing import List

import sqlalchemy
from langchain_core.documents import Document

from langchain_community.vectorstores.tidb_vector import TiDBVector
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

TiDB_CONNECT_URL = os.environ.get(
    "TEST_TiDB_VECTOR_URL", "mysql+pymysql://root@127.0.0.1:4000/test"
)

ADA_TOKEN_COUNT = 1536


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


def test_search() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = ["1", "2", "3"]
    docsearch = TiDBVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        ids=ids,
        pre_delete_collection=True,
        distance_strategy="cosine",
    )

    with docsearch._make_session() as session:
        records = list(session.query(docsearch._table_model).all())
        assert len([record.id for record in records]) == 3  # type: ignore
        session.close()

    output = docsearch.similarity_search("foo", k=1)
    docsearch.drop_table()
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_search_with_score() -> None:
    """Test end to end construction, search"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TiDBVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        pre_delete_collection=True,
        distance_strategy="cosine",
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    docsearch.drop_table()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_load_from_existing_collection() -> None:
    """Test loading existing collection."""

    # load from non-existing collection
    try:
        _ = TiDBVector.from_existing_collection(
            collection_name="test_collection_non_existing",
            embedding=FakeEmbeddingsWithAdaDimension(),
            connection_string=TiDB_CONNECT_URL,
        )
        assert False, "non-existing collection testing raised an error"
    except sqlalchemy.exc.NoSuchTableError:
        pass

    # create collection and add documents
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TiDBVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        pre_delete_collection=True,
        distance_strategy="cosine",
    )

    # load from existing collection
    docsearch_copy = TiDBVector.from_existing_collection(
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
    )
    output = docsearch_copy.similarity_search_with_score("foo", k=1)
    docsearch.drop_table()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_delete_doc() -> None:
    """Test delete document from TiDB Vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = ["1", "2", "3"]
    docsearch = TiDBVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        ids=ids,
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_score("foo", k=1)
    docsearch.delete(["1"])
    output_after_deleted = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0)]
    assert output_after_deleted == [
        (Document(page_content="bar", metadata={"page": "1"}), 0.00032557383249343097)
    ]


def test_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch_consine = TiDBVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        distance_strategy="cosine",
        pre_delete_collection=True,
    )

    output_consine = docsearch_consine.similarity_search_with_relevance_scores(
        "foo", k=3
    )
    assert output_consine == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675066),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
    ]

    docsearch_l2 = TiDBVector.from_existing_collection(
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        distance_strategy="l2",
    )
    output_l2 = docsearch_l2.similarity_search_with_relevance_scores("foo", k=3)
    assert output_l2 == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.29289321881345254),
        (Document(page_content="baz", metadata={"page": "2"}), -0.4142135623730949),
    ]

    try:
        _ = TiDBVector.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            connection_string=TiDB_CONNECT_URL,
            metadatas=metadatas,
            distance_strategy="inner",
            pre_delete_collection=True,
        )
        assert False, "inner product should raise error"
    except ValueError:
        pass

    docsearch_l2.drop_table()


def test_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TiDBVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=TiDB_CONNECT_URL,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = retriever.get_relevant_documents("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]
