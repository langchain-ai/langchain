"""Integration tests for MariaDB vector store.

These tests require a running MariaDB instance. Configure connection via:
- MARIADB_HOST (default: localhost)
- MARIADB_PORT (default: 3306)
- MARIADB_USER (default: root)
- MARIADB_PASSWORD (default: password)
- MARIADB_DATABASE (default: test_vectorstore)
"""

import os
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Skip all tests if async-mariadb-connector not available
pytest.importorskip("async_mariadb_connector")


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with fake vectors."""
        return [[float(i) for i in range(10)] for _ in texts]

    def embed_query(self, _text: str) -> list[float]:
        """Embed query with fake vector."""
        return [float(i) for i in range(10)]


@pytest.fixture
async def mariadb_connection() -> Any:
    """Create test MariaDB connection."""
    from async_mariadb_connector import AsyncMariaDB

    connection = AsyncMariaDB(
        host=os.getenv("MARIADB_HOST", "localhost"),
        port=int(os.getenv("MARIADB_PORT", "3306")),
        user=os.getenv("MARIADB_USER", "root"),
        password=os.getenv("MARIADB_PASSWORD", "password"),
        database=os.getenv("MARIADB_DATABASE", "test_vectorstore"),
        pool_size=5,
    )

    # Connect
    await connection.connect()

    yield connection

    # Cleanup
    await connection.disconnect()


@pytest.fixture
async def vectorstore(mariadb_connection: Any) -> Any:
    """Create test vector store."""
    from langchain_community.vectorstores import MariaDB

    store = MariaDB(
        connection=mariadb_connection,
        embedding_function=FakeEmbeddings(),
        table_name="test_langchain_vectors",
        create_table=True,
    )

    # Wait for table creation
    import asyncio

    await asyncio.sleep(0.5)

    yield store

    # Cleanup - drop test table
    await mariadb_connection.execute("DROP TABLE IF EXISTS test_langchain_vectors")


@pytest.mark.asyncio
class TestMariaDBVectorStore:
    """Test suite for MariaDB vector store."""

    async def test_add_texts(self, vectorstore: Any) -> None:
        """Test adding texts to vector store."""
        texts = ["foo", "bar", "baz"]
        ids = await vectorstore.aadd_texts(texts)

        assert len(ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in ids)

    async def test_add_documents(self, vectorstore: Any) -> None:
        """Test adding documents to vector store."""
        docs = [
            Document(page_content="foo", metadata={"source": "test1"}),
            Document(page_content="bar", metadata={"source": "test2"}),
        ]
        ids = await vectorstore.aadd_documents(docs)

        assert len(ids) == 2

    async def test_add_texts_with_metadata(self, vectorstore: Any) -> None:
        """Test adding texts with metadata."""
        texts = ["foo", "bar"]
        metadatas = [{"page": 1}, {"page": 2}]
        ids = await vectorstore.aadd_texts(texts, metadatas=metadatas)

        assert len(ids) == 2

    async def test_similarity_search(self, vectorstore: Any) -> None:
        """Test similarity search."""
        texts = ["foo", "bar", "baz"]
        await vectorstore.aadd_texts(texts)

        results = await vectorstore.asimilarity_search("foo", k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    async def test_similarity_search_with_score(self, vectorstore: Any) -> None:
        """Test similarity search with scores."""
        texts = ["foo", "bar", "baz"]
        await vectorstore.aadd_texts(texts)

        results = await vectorstore.asimilarity_search_with_score("foo", k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        assert all(0 <= score <= 1 for _, score in results)

    async def test_hybrid_search(self, vectorstore: Any) -> None:
        """Test hybrid search (FULLTEXT + vector)."""
        texts = [
            "MariaDB is a fast database",
            "PostgreSQL is also popular",
            "MySQL is related to MariaDB",
        ]
        await vectorstore.aadd_texts(texts)

        results = await vectorstore.ahybrid_search("MariaDB database", k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
        # Check that metadata includes scores
        assert all("combined_score" in doc.metadata for doc in results)

    async def test_delete(self, vectorstore: Any) -> None:
        """Test deleting documents."""
        texts = ["foo", "bar", "baz"]
        ids = await vectorstore.aadd_texts(texts)

        # Delete first document
        await vectorstore.adelete([ids[0]])

        # Search should return fewer results
        results = await vectorstore.asimilarity_search("foo", k=10)
        assert len(results) == 2

    async def test_from_texts(self, mariadb_connection: Any) -> None:
        """Test creating vector store from texts."""
        from langchain_community.vectorstores import MariaDB

        texts = ["foo", "bar"]
        store = await MariaDB.afrom_texts(
            texts,
            FakeEmbeddings(),
            connection=mariadb_connection,
            table_name="test_from_texts",
        )

        assert store is not None
        results = await store.asimilarity_search("foo", k=1)
        assert len(results) == 1

        # Cleanup
        await mariadb_connection.execute("DROP TABLE IF EXISTS test_from_texts")

    async def test_from_documents(self, mariadb_connection: Any) -> None:
        """Test creating vector store from documents."""
        from langchain_community.vectorstores import MariaDB

        docs = [
            Document(page_content="foo", metadata={"source": "test1"}),
            Document(page_content="bar", metadata={"source": "test2"}),
        ]
        store = await MariaDB.afrom_documents(
            docs,
            FakeEmbeddings(),
            connection=mariadb_connection,
            table_name="test_from_docs",
        )

        assert store is not None
        results = await store.asimilarity_search("foo", k=1)
        assert len(results) == 1
        assert "source" in results[0].metadata

        # Cleanup
        await mariadb_connection.execute("DROP TABLE IF EXISTS test_from_docs")

    async def test_sync_methods(self, mariadb_connection: Any) -> None:
        """Test synchronous wrapper methods."""
        from langchain_community.vectorstores import MariaDB

        store = MariaDB(
            connection=mariadb_connection,
            embedding_function=FakeEmbeddings(),
            table_name="test_sync_methods",
        )

        import asyncio

        await asyncio.sleep(0.5)

        # Test sync add_texts
        texts = ["foo", "bar"]
        ids = store.add_texts(texts)
        assert len(ids) == 2

        # Test sync similarity_search
        results = store.similarity_search("foo", k=1)
        assert len(results) == 1

        # Test sync delete
        store.delete([ids[0]])

        # Cleanup
        await mariadb_connection.execute("DROP TABLE IF EXISTS test_sync_methods")

    async def test_custom_column_names(self, mariadb_connection: Any) -> None:
        """Test using custom column names."""
        from langchain_community.vectorstores import MariaDB

        store = MariaDB(
            connection=mariadb_connection,
            embedding_function=FakeEmbeddings(),
            table_name="test_custom_cols",
            content_column="text_content",
            embedding_column="vector_data",
            metadata_column="meta_info",
        )

        import asyncio

        await asyncio.sleep(0.5)

        texts = ["foo"]
        ids = await store.aadd_texts(texts)
        assert len(ids) == 1

        # Cleanup
        await mariadb_connection.execute("DROP TABLE IF EXISTS test_custom_cols")

    async def test_large_batch_insert(self, vectorstore: Any) -> None:
        """Test inserting large batch of documents."""
        texts = [f"Document {i}" for i in range(100)]
        ids = await vectorstore.aadd_texts(texts)

        assert len(ids) == 100

        results = await vectorstore.asimilarity_search("Document", k=10)
        assert len(results) == 10

    async def test_metadata_persistence(self, vectorstore: Any) -> None:
        """Test that metadata is correctly persisted and retrieved."""
        texts = ["test content"]
        metadatas = [{"author": "John", "page": 42, "tags": ["ml", "ai"]}]
        await vectorstore.aadd_texts(texts, metadatas=metadatas)

        results = await vectorstore.asimilarity_search("test", k=1)
        assert len(results) == 1
        assert results[0].metadata["author"] == "John"
        assert results[0].metadata["page"] == 42
        assert results[0].metadata["tags"] == ["ml", "ai"]


@pytest.mark.asyncio
class TestMariaDBPerformance:
    """Performance tests for MariaDB vector store."""

    async def test_bulk_insert_performance(self, vectorstore: Any) -> None:
        """Test bulk insert performance."""
        import time

        texts = [f"Document {i} with some content" for i in range(1000)]

        start = time.time()
        ids = await vectorstore.aadd_texts(texts)
        elapsed = time.time() - start

        assert len(ids) == 1000
        # Should be able to insert 1000 docs in under 1 second
        assert elapsed < 1.0

    async def test_search_performance(self, vectorstore: Any) -> None:
        """Test search performance."""
        import time

        # Insert documents
        texts = [f"Document {i} about database technology" for i in range(100)]
        await vectorstore.aadd_texts(texts)

        # Test search performance
        start = time.time()
        results = await vectorstore.asimilarity_search("database", k=10)
        elapsed = time.time() - start

        assert len(results) == 10
        # Search should be fast (< 100ms for 100 docs)
        assert elapsed < 0.1

    async def test_hybrid_search_performance(self, vectorstore: Any) -> None:
        """Test hybrid search performance."""
        import time

        # Insert documents with varied content
        texts = [f"MariaDB document {i} about databases" for i in range(50)] + [
            f"Python document {i} about programming" for i in range(50)
        ]
        await vectorstore.aadd_texts(texts)

        # Test hybrid search performance
        start = time.time()
        results = await vectorstore.ahybrid_search("MariaDB database", k=10)
        elapsed = time.time() - start

        assert len(results) == 10
        # Hybrid search should be reasonably fast
        assert elapsed < 0.2
