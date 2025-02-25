"""Test MariaDBStore functionality."""

import contextlib
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Sequence

import pytest
import sqlalchemy
from langchain_core.documents import Document
from sqlalchemy import Engine, create_engine

from langchain_community.vectorstores.mariadb import (
    MariaDBStore,
    MariaDBStoreSettings,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from tests.integration_tests.vectorstores.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
    TYPE_5_FILTERING_TEST_CASES,
)

USER = os.environ.get("MARIADB_USER", "langchain")
PASSWORD = os.environ.get("MARIADB_PASSWORD", "langchain")
HOST = os.environ.get("MARIADB_HOST", "localhost")
PORT = int(os.environ.get("MARIADB_PORT", "3306"))
DB = os.environ.get("MARIADB_DATABASE", "langchain")

URL = f"mariadb+mariadbconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"


def url() -> str:
    return URL


@contextmanager
def pool() -> Generator[Engine, None, None]:
    # Establish a connection to your test database
    engine = create_engine(url=URL)
    try:
        yield engine
    finally:
        # Cleanup: close the pool after the test is done
        con = engine.raw_connection()
        cursor = con.cursor()
        try:
            cursor.execute("DROP TABLE IF EXISTS langchain_embedding")
            cursor.execute("DROP TABLE IF EXISTS langchain_collection")
        except Exception:
            pass
        cursor.close()
        engine.dispose()


ADA_TOKEN_COUNT = 1536


def _compare_documents(left: Sequence[Document], right: Sequence[Document]) -> None:
    """Compare lists of documents, irrespective of IDs."""
    assert len(left) == len(right)
    for left_doc, right_doc in zip(left, right):
        assert left_doc.page_content == right_doc.page_content
        assert left_doc.metadata == right_doc.metadata


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


def test_mariadbstore_with_pool() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    engine = create_engine(url=url())

    docsearch = MariaDBStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        embedding_length=ADA_TOKEN_COUNT,
        datasource=engine,
        config=MariaDBStoreSettings(pre_delete_collection=True),
    )
    output = docsearch.similarity_search("foo", k=1)
    _compare_documents(output, [Document(page_content="foo")])

    output = docsearch.search("foo", "similarity", k=1)
    _compare_documents(output, [Document(page_content="foo")])

    engine.dispose()


def test_mariadbstore_with_url() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    url_value = url()
    docsearch = MariaDBStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        embedding_length=ADA_TOKEN_COUNT,
        datasource=url_value,
        engine_args={"pool_size": 2},
        config=MariaDBStoreSettings(pre_delete_collection=True),
    )
    output = docsearch.similarity_search("foo", k=1)
    _compare_documents(output, [Document(page_content="foo")])

    output = docsearch.search("foo", "similarity", k=1)
    _compare_documents(output, [Document(page_content="foo")])


def test_mariadbstore_with_sqlalchemy() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    url_value = url()
    docsearch = MariaDBStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        embedding_length=ADA_TOKEN_COUNT,
        datasource=sqlalchemy.create_engine(url_value),
        config=MariaDBStoreSettings(pre_delete_collection=True),
    )
    output = docsearch.similarity_search("foo", k=1)
    _compare_documents(output, [Document(page_content="foo")])

    output = docsearch.search("foo", "similarity", k=1)
    _compare_documents(output, [Document(page_content="foo")])


@pytest.mark.asyncio
async def test_amariadbstore() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            embedding_length=ADA_TOKEN_COUNT,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        _compare_documents(output, [Document(page_content="foo")])

        output = await docsearch.asearch("foo", "similarity", k=1)
        _compare_documents(output, [Document(page_content="foo")])


def test_mariadb_store_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    with pool() as tmppool:
        docsearch = MariaDBStore.from_embeddings(
            text_embeddings=text_embedding_pairs,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.similarity_search("foo", k=1)
        _compare_documents(output, [Document(page_content="foo")])


@pytest.mark.asyncio
async def test_amariadb_store_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    with pool() as tmppool:
        docsearch = MariaDBStore.from_embeddings(
            text_embeddings=text_embedding_pairs,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        _compare_documents(output, [Document(page_content="foo")])


def test_mariadb_store_embeddings_config() -> None:
    store_config = MariaDBStoreSettings()
    store_config.pre_delete_collection = True
    store_config.tables.embedding_table = "emb_table"
    store_config.tables.collection_table = "col_table"

    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    with pool() as tmppool:
        docsearch = MariaDBStore.from_embeddings(
            text_embeddings=text_embedding_pairs,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.similarity_search("foo", k=1)
        _compare_documents(output, [Document(page_content="foo")])


@pytest.mark.asyncio
async def test_amariadb_store_embeddings_config() -> None:
    store_config = MariaDBStoreSettings()
    store_config.pre_delete_collection = True
    store_config.tables.embedding_table = "emb_table"
    store_config.tables.collection_table = "col_table"

    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    with pool() as tmppool:
        docsearch = MariaDBStore.from_embeddings(
            text_embeddings=text_embedding_pairs,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        _compare_documents(output, [Document(page_content="foo")])


def test_mariadb_store_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.similarity_search("foo", k=1)
        _compare_documents(
            output, [Document(page_content="foo", metadata={"page": "0"})]
        )


@pytest.mark.asyncio
async def test_amariadb_store_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        _compare_documents(
            output, [Document(page_content="foo", metadata={"page": "0"})]
        )


def test_mariadb_store_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.similarity_search_with_score("foo", k=1)
        docs, scores = zip(*output)
        _compare_documents(docs, [Document(page_content="foo", metadata={"page": "0"})])
        assert scores == (0.0,)


@pytest.mark.asyncio
async def test_amariadb_store_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.asimilarity_search_with_score("foo", k=1)
        docs, scores = zip(*output)
        _compare_documents(docs, [Document(page_content="foo", metadata={"page": "0"})])
        assert scores == (0.0,)


def test_mariadb_store_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.similarity_search_with_score(
            "foo", k=1, filter={"page": "0"}
        )
        docs, scores = zip(*output)
        _compare_documents(docs, [Document(page_content="foo", metadata={"page": "0"})])
        assert scores == (0.0,)


@pytest.mark.asyncio
async def test_amariadb_store_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.asimilarity_search_with_score(
            "foo", k=1, filter={"page": "0"}
        )
        docs, scores = zip(*output)
        _compare_documents(docs, [Document(page_content="foo", metadata={"page": "0"})])
        assert scores == (0.0,)


def test_mariadb_store_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.similarity_search_with_score(
            "foo", k=1, filter={"page": "2"}
        )
        docs, scores = zip(*output)
        _compare_documents(docs, [Document(page_content="baz", metadata={"page": "2"})])
        assert scores == (0.0013003906671379406,)


def test_mariadb_store_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.similarity_search_with_score(
            "foo", k=1, filter={"page": "5"}
        )
        assert output == []


def test_mariadb_store_collection_with_metadata() -> None:
    """Test end to end collection construction"""
    with pool() as tmppool:
        con = tmppool.raw_connection()
        cursor = con.cursor()
        try:
            cursor.execute("TRUNCATE TABLE langchain_embedding")
            cursor.execute("DELETE FROM langchain_collection")
            con.commit()
        except Exception:
            pass
        cursor.close()
        con.close()

        MariaDBStore(
            embeddings=FakeEmbeddingsWithAdaDimension(),
            collection_name="test_collection",
            collection_metadata={"foo": "bar"},
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )

        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute("SELECT label,metadata  FROM langchain_collection")
        row = cursor.fetchone()
        if row is None:
            assert False, "Expected a collection to exists but received None"
        else:
            assert row[0] == "test_collection"
            assert row[1] == '{"foo": "bar"}'
        cursor.close()
        con.close()


def test_mariadb_get_by_ids_format() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        vectorstore = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            ids=[
                "00000000-0000-4000-0000-000000000000",
                "10000000-0000-4000-0000-000000000000",
                "20000000-0000-4000-0000-000000000000",
            ],
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        ids = vectorstore.add_documents(documents, ids=["1", "2"])

        retrieved_documents = vectorstore.get_by_ids(ids)
        assert retrieved_documents == [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

        # Id format can only be UUID or numeric format
        try:
            vectorstore.add_documents(documents, ids=["1/'1", "2"])
            raise RuntimeError("must have thrown an error")
        except ValueError:
            pass
        try:
            vectorstore.get_by_ids(["1/'1", "2"])
            raise RuntimeError("must have thrown an error")
        except ValueError:
            pass

        retrieved_documents = vectorstore.get_by_ids([])
        assert retrieved_documents == []

        retrieved_documents = vectorstore.get_by_ids(["blou"])
        assert retrieved_documents == []


@pytest.mark.asyncio
async def test_amariadb_get_by_ids_format() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        vectorstore = await MariaDBStore.afrom_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            ids=[
                "00000000-0000-4000-0000-000000000000",
                "10000000-0000-4000-0000-000000000000",
                "20000000-0000-4000-0000-000000000000",
            ],
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        ids = await vectorstore.aadd_documents(documents, ids=["1", "2"])

        retrieved_documents = await vectorstore.aget_by_ids(ids)
        assert retrieved_documents == [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

        # Id format can only be UUID or numeric format
        try:
            await vectorstore.aadd_documents(documents, ids=["1/'1", "2"])
            raise RuntimeError("must have thrown an error")
        except ValueError:
            pass
        try:
            await vectorstore.aget_by_ids(["1/'1", "2"])
            raise RuntimeError("must have thrown an error")
        except ValueError:
            pass

        retrieved_documents = await vectorstore.aget_by_ids([])
        assert retrieved_documents == []

        retrieved_documents = await vectorstore.aget_by_ids(["blou"])
        assert retrieved_documents == []


def test_mariadb_store_delete_docs() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        vectorstore = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            ids=[
                "00000000-0000-4000-0000-000000000000",
                "10000000-0000-4000-0000-000000000000",
                "20000000-0000-4000-0000-000000000000",
            ],
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        vectorstore.delete(
            [
                "00000000-0000-4000-0000-000000000000",
                "10000000-0000-4000-0000-000000000000",
            ]
        )
        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute("SELECT id FROM langchain_embedding")
        rows = cursor.fetchall()
        cursor.close()
        con.close()

        assert len(rows) == 1
        assert rows[0][0] == "20000000-0000-4000-0000-000000000000"

        vectorstore.delete(
            [
                "10000000-0000-4000-0000-000000000000",
                "20000000-0000-4000-0000-000000000000",
            ]
        )  # Should not raise on missing ids
        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute("SELECT id FROM langchain_embedding")
        rows = cursor.fetchall()
        cursor.close()
        con.close()
        assert len(rows) == 0


@pytest.mark.asyncio
async def test_amariadb_store_delete_docs() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        vectorstore = await MariaDBStore.afrom_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            ids=[
                "00000000-0000-4000-0000-000000000000",
                "10000000-0000-4000-0000-000000000000",
                "20000000-0000-4000-0000-000000000000",
            ],
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        await vectorstore.adelete(
            [
                "00000000-0000-4000-0000-000000000000",
                "10000000-0000-4000-0000-000000000000",
            ]
        )
        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute("SELECT id FROM langchain_embedding")
        rows = cursor.fetchall()
        cursor.close()
        con.close()
        assert len(rows) == 1
        assert rows[0][0] == "20000000-0000-4000-0000-000000000000"

        await vectorstore.adelete(
            [
                "10000000-0000-4000-0000-000000000000",
                "20000000-0000-4000-0000-000000000000",
            ]
        )  # Should not raise on missing ids
        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute("SELECT id FROM langchain_embedding")
        rows = cursor.fetchall()
        cursor.close()
        con.close()
        assert len(rows) == 0


def test_mariadb_store_delete_collection() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        vectorstore = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            ids=[
                "00000000-0000-4000-0000-000000000000",
                "10000000-0000-4000-0000-000000000000",
                "20000000-0000-4000-0000-000000000000",
            ],
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        vectorstore.delete(collection_only=True)


def test_mariadb_store_index_documents() -> None:
    """Test adding duplicate documents results in overwrites."""
    documents = [
        Document(
            page_content="there are cats in the pond",
            metadata={"id": 1, "location": "pond", "topic": "animals"},
            id="10000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"id": 2, "location": "pond", "topic": "animals"},
            id="20000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="fresh apples are available at the market",
            metadata={"id": 3, "location": "market", "topic": "food"},
            id="30000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="the market also sells fresh oranges",
            metadata={"id": 4, "location": "market", "topic": "food"},
            id="40000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="the new art exhibit is fascinating",
            metadata={"id": 5, "location": "museum", "topic": "art"},
            id="50000000-0000-4000-0000-000000000000",
        ),
    ]
    ids: List[str] = [value for doc in documents if (value := doc.id) is not None]
    with pool() as tmppool:
        vectorstore = MariaDBStore.from_documents(
            documents=documents,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            ids=ids,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute(
            "SELECT e.id FROM langchain_embedding e "
            "LEFT JOIN langchain_collection c ON e.collection_id = c.id "
            "WHERE label = 'test_collection_filter' order by id"
        )
        rows = cursor.fetchall()
        cursor.close()
        con.close()
        assert sorted(record[0] for record in rows) == [
            "10000000-0000-4000-0000-000000000000",
            "20000000-0000-4000-0000-000000000000",
            "30000000-0000-4000-0000-000000000000",
            "40000000-0000-4000-0000-000000000000",
            "50000000-0000-4000-0000-000000000000",
        ]

        # Try to overwrite the first document
        documents = [
            Document(
                page_content="new content in the zoo",
                metadata={"id": 1, "location": "zoo", "topic": "zoo"},
                id="10000000-0000-4000-0000-000000000000",
            ),
        ]

        vectorstore.add_documents(documents, ids=[doc.id for doc in documents])

        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute(
            "SELECT e.id, e.metadata FROM langchain_embedding e "
            "LEFT JOIN langchain_collection c ON e.collection_id = c.id "
            "WHERE label = 'test_collection_filter' order by id"
        )
        rows = cursor.fetchall()
        cursor.close()
        con.close()
        assert sorted(record[0] for record in rows) == [
            "10000000-0000-4000-0000-000000000000",
            "20000000-0000-4000-0000-000000000000",
            "30000000-0000-4000-0000-000000000000",
            "40000000-0000-4000-0000-000000000000",
            "50000000-0000-4000-0000-000000000000",
        ]

        assert json.loads(rows[0][1]) == {
            "id": 1,
            "location": "zoo",
            "topic": "zoo",
        }


@pytest.mark.asyncio
async def test_amariadb_store_index_documents() -> None:
    """Test adding duplicate documents results in overwrites."""
    documents = [
        Document(
            page_content="there are cats in the pond",
            metadata={"id": 1, "location": "pond", "topic": "animals"},
            id="10000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"id": 2, "location": "pond", "topic": "animals"},
            id="20000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="fresh apples are available at the market",
            metadata={"id": 3, "location": "market", "topic": "food"},
            id="30000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="the market also sells fresh oranges",
            metadata={"id": 4, "location": "market", "topic": "food"},
            id="40000000-0000-4000-0000-000000000000",
        ),
        Document(
            page_content="the new art exhibit is fascinating",
            metadata={"id": 5, "location": "museum", "topic": "art"},
            id="50000000-0000-4000-0000-000000000000",
        ),
    ]
    ids: List[str] = [value for doc in documents if (value := doc.id) is not None]
    with pool() as tmppool:
        vectorstore = await MariaDBStore.afrom_documents(
            documents=documents,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            ids=ids,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute(
            "SELECT e.id FROM langchain_embedding e "
            "LEFT JOIN langchain_collection c ON e.collection_id = c.id "
            "WHERE label = 'test_collection_filter' order by id"
        )
        rows = cursor.fetchall()
        assert sorted(record[0] for record in rows) == [
            "10000000-0000-4000-0000-000000000000",
            "20000000-0000-4000-0000-000000000000",
            "30000000-0000-4000-0000-000000000000",
            "40000000-0000-4000-0000-000000000000",
            "50000000-0000-4000-0000-000000000000",
        ]
        cursor.close()
        con.close()

        # Try to overwrite the first document
        documents = [
            Document(
                page_content="new content in the zoo",
                metadata={"id": 1, "location": "zoo", "topic": "zoo"},
                id="10000000-0000-4000-0000-000000000000",
            ),
        ]

        await vectorstore.aadd_documents(documents, ids=[doc.id for doc in documents])
        con = tmppool.raw_connection()
        cursor = con.cursor()
        cursor.execute(
            "SELECT e.id, e.metadata FROM langchain_embedding e "
            "LEFT JOIN langchain_collection c ON e.collection_id = c.id "
            "WHERE label = 'test_collection_filter' order by id"
        )
        rows = cursor.fetchall()
        assert sorted(record[0] for record in rows) == [
            "10000000-0000-4000-0000-000000000000",
            "20000000-0000-4000-0000-000000000000",
            "30000000-0000-4000-0000-000000000000",
            "40000000-0000-4000-0000-000000000000",
            "50000000-0000-4000-0000-000000000000",
        ]

        assert json.loads(rows[0][1]) == {
            "id": 1,
            "location": "zoo",
            "topic": "zoo",
        }
        cursor.close()
        con.close()


def test_mariadb_store_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )

        output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
        docs, scores = zip(*output)
        _compare_documents(
            docs,
            [
                Document(page_content="foo", metadata={"page": "0"}),
                Document(page_content="bar", metadata={"page": "1"}),
                Document(page_content="baz", metadata={"page": "2"}),
            ],
        )
        assert scores == (1.0, 0.9996744261675065, 0.9986996093328621)


@pytest.mark.asyncio
async def test_amariadb_store_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = await MariaDBStore.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )

        output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)
        docs, scores = zip(*output)
        _compare_documents(
            docs,
            [
                Document(page_content="foo", metadata={"page": "0"}),
                Document(page_content="bar", metadata={"page": "1"}),
                Document(page_content="baz", metadata={"page": "2"}),
            ],
        )
        assert scores == (1.0, 0.9996744261675065, 0.9986996093328621)


def test_mariadb_store_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )

        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.999},
        )
        output = retriever.get_relevant_documents("summer")
        _compare_documents(
            output,
            [
                Document(page_content="foo", metadata={"page": "0"}),
                Document(page_content="bar", metadata={"page": "1"}),
            ],
        )


@pytest.mark.asyncio
async def test_amariadb_store_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = await MariaDBStore.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )

        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.999},
        )
        output = await retriever.aget_relevant_documents("summer")
        _compare_documents(
            output,
            [
                Document(page_content="foo", metadata={"page": "0"}),
                Document(page_content="bar", metadata={"page": "1"}),
            ],
        )


def test_mariadb_store_retriever_search_threshold_custom_normalization_fn() -> None:
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
            relevance_score_fn=lambda d: d * 0,
        )

        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5},
        )
        output = retriever.get_relevant_documents("foo")
        assert output == []


def test_mariadb_store_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
        _compare_documents(output, [Document(page_content="foo")])


@pytest.mark.asyncio
async def test_amariadb_store_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    with pool() as tmppool:
        docsearch = await MariaDBStore.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.amax_marginal_relevance_search("foo", k=1, fetch_k=3)
        _compare_documents(output, [Document(page_content="foo")])


def test_mariadb_store_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    with pool() as tmppool:
        docsearch = MariaDBStore.from_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = docsearch.max_marginal_relevance_search_with_score(
            "foo", k=1, fetch_k=3
        )
        docs, scores = zip(*output)
        _compare_documents(docs, [Document(page_content="foo")])
        assert scores == (0.0,)


@pytest.mark.asyncio
async def test_amariadb_store_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    with pool() as tmppool:
        docsearch = await MariaDBStore.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
        )
        output = await docsearch.amax_marginal_relevance_search_with_score(
            "foo", k=1, fetch_k=3
        )
        docs, scores = zip(*output)
        _compare_documents(docs, [Document(page_content="foo")])
        assert scores == (0.0,)


@contextlib.contextmanager
def get_vectorstore() -> Generator[MariaDBStore, None, None]:
    """Get a pre-populated-vectorstore"""
    with pool() as tmppool:
        store = MariaDBStore.from_documents(
            documents=DOCUMENTS,
            embedding=FakeEmbeddingsWithAdaDimension(),
            collection_name="test_collection",
            datasource=tmppool,
            config=MariaDBStoreSettings(pre_delete_collection=True),
            relevance_score_fn=lambda d: d * 0,
        )
        try:
            yield store
        finally:
            store.drop_tables()


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_1_FILTERING_TEST_CASES)
def test_mariadb_store_with_with_metadata_filters_1(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    with get_vectorstore() as store:
        docs = store.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_2_FILTERING_TEST_CASES)
def test_mariadb_store_with_with_metadata_filters_2(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    with get_vectorstore() as store:
        docs = store.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_3_FILTERING_TEST_CASES)
def test_mariadb_store_with_with_metadata_filters_3(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    with get_vectorstore() as store:
        docs = store.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_4_FILTERING_TEST_CASES)
def test_mariadb_store_with_with_metadata_filters_4(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    with get_vectorstore() as store:
        docs = store.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_5_FILTERING_TEST_CASES)
def test_mariadb_store_with_with_metadata_filters_5(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    with get_vectorstore() as store:
        docs = store.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter
