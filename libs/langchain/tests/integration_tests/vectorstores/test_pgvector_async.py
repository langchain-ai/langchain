"""Test PGVector functionality."""
import os
from contextlib import asynccontextmanager
from typing import List

import pytest
from sqlalchemy import select

from langchain.docstore.document import Document
from langchain.vectorstores.pgvector_async import PGVectorAsync
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

DRIVER = os.environ.get("TEST_PGVECTOR_DRIVER", "asyncpg")
HOST = os.environ.get("TEST_PGVECTOR_HOST", "localhost")
PORT = int(os.environ.get("TEST_PGVECTOR_PORT", "5432"))
DATABASE = os.environ.get("TEST_PGVECTOR_DATABASE", "postgres")
USER = os.environ.get("TEST_PGVECTOR_USER", "postgres")
PASSWORD = os.environ.get("TEST_PGVECTOR_PASSWORD", "postgres")

DATABASE_URL = f"postgresql+{DRIVER}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

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


@asynccontextmanager
async def with_db():
    vectorstore = PGVectorAsync(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        db_url=DATABASE_URL,
    )

    await vectorstore.drop_schema()
    await vectorstore.create_schema()
    yield
    await vectorstore.drop_schema()


@pytest.mark.asyncio
async def test_pgvector() -> None:
    """Test end to end construction and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_pgvector_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
        text_embedding_pairs = list(zip(texts, text_embeddings))
        docsearch = await PGVectorAsync.afrom_embeddings(
            text_embeddings=text_embedding_pairs,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_pgvector_with_metadatas() -> None:
    """Test end to end construction and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": "0"})]


@pytest.mark.asyncio
async def test_pgvector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search_with_score("foo", k=1)
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


@pytest.mark.asyncio
async def test_pgvector_with_filter_match() -> None:
    """Test end to end construction and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search_with_score(
            "foo", k=1, filter={"page": "0"}
        )
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


@pytest.mark.asyncio
async def test_pgvector_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search_with_score(
            "foo", k=1, filter={"page": "2"}
        )
        assert output == [
            (
                Document(page_content="baz", metadata={"page": "2"}),
                0.0013003906671379406,
            )
        ]


@pytest.mark.asyncio
async def test_pgvector_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search_with_score(
            "foo", k=1, filter={"page": "5"}
        )
        assert output == []


@pytest.mark.asyncio
async def test_pgvector_collection_with_metadata() -> None:
    """Test end to end collection construction"""
    async with with_db():
        pgvector = PGVectorAsync(
            collection_name="test_collection",
            collection_metadata={"foo": "bar"},
            embeddings=FakeEmbeddingsWithAdaDimension(),
            db_url=DATABASE_URL,
        )
        await pgvector.delete_collection()  # Delete collection if it exists
        await pgvector.create_collection()
        collection = await pgvector.get_collection()
        if collection is None:
            assert False, "Expected a CollectionStore object but received None"
        else:
            assert collection.name == "test_collection"
            assert collection.cmetadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_pgvector_with_filter_in_set() -> None:
    """Test end to end construction and search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.asimilarity_search_with_score(
            "foo", k=2, filter={"page": {"IN": ["0", "2"]}}
        )
        assert output == [
            (Document(page_content="foo", metadata={"page": "0"}), 0.0),
            (
                Document(page_content="baz", metadata={"page": "2"}),
                0.0013003906671379406,
            ),
        ]


@pytest.mark.asyncio
async def test_pgvector_delete_docs() -> None:
    """Add and delete documents."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection_filter",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            ids=["1", "2", "3"],
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        await docsearch.adelete(["1", "2"])
        async with docsearch._make_session() as session:
            query = select(docsearch.EmbeddingStore)
            results = await session.execute(query)
            records = list(results.scalars().all())

            assert sorted(record.custom_id for record in records) == ["3"]

        await docsearch.adelete(["2", "3"])  # Should not raise on missing ids
        async with docsearch._make_session() as session:
            query = select(docsearch.EmbeddingStore)
            results = await session.execute(query)
            records = list(results.scalars().all())

            assert sorted(record.custom_id for record in records) == []  # type: ignore


@pytest.mark.asyncio
async def test_pgvector_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )

        output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)
        assert output == [
            (Document(page_content="foo", metadata={"page": "0"}), 1.0),
            (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675065),
            (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
        ]


@pytest.mark.asyncio
async def test_pgvector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )

        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.999},
        )
        output = await retriever.aget_relevant_documents("summer")
        assert output == [
            Document(page_content="foo", metadata={"page": "0"}),
            Document(page_content="bar", metadata={"page": "1"}),
        ]


@pytest.mark.asyncio
async def test_pgvector_retriever_search_threshold_custom_normalization_fn() -> None:
    """Test searching with threshold and custom normalization function"""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            metadatas=metadatas,
            db_url=DATABASE_URL,
            pre_delete_collection=True,
            relevance_score_fn=lambda d: d * 0,
        )

        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5},
        )
        output = await retriever.aget_relevant_documents("foo")
        assert output == []


@pytest.mark.asyncio
async def test_pgvector_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.amax_marginal_relevance_search("foo", k=1, fetch_k=3)
        assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_pgvector_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    async with with_db():
        texts = ["foo", "bar", "baz"]
        docsearch = await PGVectorAsync.afrom_texts(
            texts=texts,
            collection_name="test_collection",
            embedding=FakeEmbeddingsWithAdaDimension(),
            db_url=DATABASE_URL,
            pre_delete_collection=True,
        )
        output = await docsearch.amax_marginal_relevance_search_with_score(
            "foo", k=1, fetch_k=3
        )
        assert output == [(Document(page_content="foo"), 0.0)]
