"""Test TiDB Vector functionality."""

import os
from typing import List

from langchain_core.documents import Document

from langchain_community.vectorstores import TiDBVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

TiDB_CONNECT_URL = os.environ.get(
    "TEST_TiDB_CONNECTION_URL", "mysql+pymysql://root@127.0.0.1:4000/test"
)

ADA_TOKEN_COUNT = 1536


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings based on ASCII values of text characters."""
        return [self._text_to_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings based on ASCII values of text characters."""
        return self._text_to_embedding(text)

    def _text_to_embedding(self, text: str) -> List[float]:
        """Convert text to a unique embedding using ASCII values."""
        ascii_values = [float(ord(char)) for char in text]
        # Pad or trim the list to make it of length ADA_TOKEN_COUNT
        ascii_values = ascii_values[:ADA_TOKEN_COUNT] + [0.0] * (
            ADA_TOKEN_COUNT - len(ascii_values)
        )
        return ascii_values


def test_search() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = ["1", "2", "3"]
    docsearch = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        ids=ids,
        drop_existing_table=True,
        distance_strategy="cosine",
    )

    with docsearch.tidb_vector_client._make_session() as session:
        records = list(session.query(docsearch.tidb_vector_client._table_model).all())
        assert len([record.id for record in records]) == 3  # type: ignore
        session.close()

    output = docsearch.similarity_search("foo", k=1)
    docsearch.drop_vectorstore()
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_search_with_filter() -> None:
    """Test end to end construction and search."""

    # no metadata
    texts = ["foo", "bar", "baz"]
    docsearch = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        drop_existing_table=True,
    )

    output = docsearch.similarity_search("foo", k=1)
    output_filtered = docsearch.similarity_search(
        "foo", k=1, filter={"filter_condition": "N/A"}
    )
    assert output == [Document(page_content="foo")]
    assert output_filtered == []

    # having metadata
    metadatas = [{"page": i + 1, "page_str": str(i + 1)} for i in range(len(texts))]
    docsearch = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        drop_existing_table=True,
    )

    output = docsearch.similarity_search("foo", k=1, filter={"page": 1})
    assert output == [
        Document(page_content="foo", metadata={"page": 1, "page_str": "1"})
    ]

    # test mismatched value
    output = docsearch.similarity_search("foo", k=1, filter={"page": "1"})
    assert output == []

    # test non-existing key
    output = docsearch.similarity_search("foo", k=1, filter={"filter_condition": "N/A"})
    assert output == []

    # test IN, NIN expression
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$in": [1, 2]}})
    assert output == [
        Document(page_content="foo", metadata={"page": 1, "page_str": "1"})
    ]
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$nin": [1, 2]}})
    assert output == [
        Document(page_content="baz", metadata={"page": 3, "page_str": "3"})
    ]
    output = docsearch.similarity_search(
        "foo", k=1, filter={"page": {"$in": ["1", "2"]}}
    )
    assert output == []
    output = docsearch.similarity_search(
        "foo", k=1, filter={"page_str": {"$in": ["1", "2"]}}
    )
    assert output == [
        Document(page_content="foo", metadata={"page": 1, "page_str": "1"})
    ]

    # test GT, GTE, LT, LTE expression
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$gt": 1}})
    assert output == [
        Document(page_content="bar", metadata={"page": 2, "page_str": "2"})
    ]
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$gte": 1}})
    assert output == [
        Document(page_content="foo", metadata={"page": 1, "page_str": "1"})
    ]
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$lt": 3}})
    assert output == [
        Document(page_content="foo", metadata={"page": 1, "page_str": "1"})
    ]
    output = docsearch.similarity_search("baz", k=1, filter={"page": {"$lte": 3}})
    assert output == [
        Document(page_content="baz", metadata={"page": 3, "page_str": "3"})
    ]
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$gt": 3}})
    assert output == []
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$lt": 1}})
    assert output == []

    # test eq, neq expression
    output = docsearch.similarity_search("foo", k=1, filter={"page": {"$eq": 3}})
    assert output == [
        Document(page_content="baz", metadata={"page": 3, "page_str": "3"})
    ]
    output = docsearch.similarity_search("bar", k=1, filter={"page": {"$ne": 2}})
    assert output == [
        Document(page_content="baz", metadata={"page": 3, "page_str": "3"})
    ]

    # test AND, OR expression
    output = docsearch.similarity_search(
        "bar", k=1, filter={"$and": [{"page": 1}, {"page_str": "1"}]}
    )
    assert output == [
        Document(page_content="foo", metadata={"page": 1, "page_str": "1"})
    ]
    output = docsearch.similarity_search(
        "bar", k=1, filter={"$or": [{"page": 1}, {"page_str": "2"}]}
    )
    assert output == [
        Document(page_content="bar", metadata={"page": 2, "page_str": "2"}),
    ]
    output = docsearch.similarity_search(
        "foo",
        k=1,
        filter={
            "$or": [{"page": 1}, {"page": 2}],
            "$and": [{"page": 2}],
        },
    )
    assert output == [
        Document(page_content="bar", metadata={"page": 2, "page_str": "2"})
    ]
    output = docsearch.similarity_search(
        "foo", k=1, filter={"$and": [{"$or": [{"page": 1}, {"page": 2}]}, {"page": 3}]}
    )
    assert output == []

    docsearch.drop_vectorstore()


def test_search_with_score() -> None:
    """Test end to end construction, search"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        drop_existing_table=True,
        distance_strategy="cosine",
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    docsearch.drop_vectorstore()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_load_from_existing_vectorstore() -> None:
    """Test loading existing TiDB Vector Store."""

    # create tidb vector store and add documents
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        drop_existing_table=True,
        distance_strategy="cosine",
    )

    # load from existing tidb vector store
    docsearch_copy = TiDBVectorStore.from_existing_vector_table(
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
    )
    output = docsearch_copy.similarity_search_with_score("foo", k=1)
    docsearch.drop_vectorstore()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]

    # load from non-existing tidb vector store
    try:
        _ = TiDBVectorStore.from_existing_vector_table(
            table_name="test_vectorstore_non_existing",
            embedding=FakeEmbeddingsWithAdaDimension(),
            connection_string=TiDB_CONNECT_URL,
        )
        assert False, "non-existing tidb vector store testing raised an error"
    except ValueError:
        pass


def test_delete_doc() -> None:
    """Test delete document from TiDB Vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = ["1", "2", "3"]
    docsearch = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        ids=ids,
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        drop_existing_table=True,
    )

    output = docsearch.similarity_search_with_score("foo", k=1)
    docsearch.delete(["1", "2"])
    output_after_deleted = docsearch.similarity_search_with_score("foo", k=1)
    docsearch.drop_vectorstore()
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0)]
    assert output_after_deleted == [
        (Document(page_content="baz", metadata={"page": "2"}), 0.004691842206844599)
    ]


def test_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch_consine = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        metadatas=metadatas,
        distance_strategy="cosine",
        drop_existing_table=True,
    )

    output_consine = docsearch_consine.similarity_search_with_relevance_scores(
        "foo", k=3
    )
    assert output_consine == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9977280385800326),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9953081577931554),
    ]

    docsearch_l2 = TiDBVectorStore.from_existing_vector_table(
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=TiDB_CONNECT_URL,
        distance_strategy="l2",
    )
    output_l2 = docsearch_l2.similarity_search_with_relevance_scores("foo", k=3)
    assert output_l2 == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), -9.51189802081432),
        (Document(page_content="baz", metadata={"page": "2"}), -11.90348790056394),
    ]

    try:
        _ = TiDBVectorStore.from_texts(
            texts=texts,
            table_name="test_tidb_vectorstore_langchain",
            embedding=FakeEmbeddingsWithAdaDimension(),
            connection_string=TiDB_CONNECT_URL,
            metadatas=metadatas,
            distance_strategy="inner",
            drop_existing_table=True,
        )
        assert False, "inner product should raise error"
    except ValueError:
        pass

    docsearch_l2.drop_vectorstore()  # type: ignore[attr-defined]


def test_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TiDBVectorStore.from_texts(
        texts=texts,
        table_name="test_tidb_vectorstore_langchain",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=TiDB_CONNECT_URL,
        drop_existing_table=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.997},
    )
    output = retriever.invoke("foo")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]

    docsearch.drop_vectorstore()
