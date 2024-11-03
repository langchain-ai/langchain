"""Test suite to test vectostores."""

from abc import abstractmethod

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding, Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_standard_tests.base import BaseStandardTests

# Arbitrarily chosen. Using a small embedding size
# so tests are faster and easier to debug.
EMBEDDING_SIZE = 6


class ReadWriteTestSuite(BaseStandardTests):
    """Test suite for checking the read-write API of a vectorstore.

    This test suite verifies the basic read-write API of a vectorstore.

    The test suite is designed for synchronous vectorstores.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty vectorstore for each test.

    The fixture should use the `get_embeddings` method to get a pre-defined
    embeddings model that should be used for this test suite.
    """

    @abstractmethod
    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        """Get the vectorstore class to test.

        The returned vectorstore should be EMPTY.
        """

    @staticmethod
    def get_embeddings() -> Embeddings:
        """A pre-defined embeddings model that should be used for this test."""
        return DeterministicFakeEmbedding(
            size=EMBEDDING_SIZE,
        )

    def test_vectorstore_is_empty(self, vectorstore: VectorStore) -> None:
        """Test that the vectorstore is empty."""
        assert vectorstore.similarity_search("foo", k=1) == []

    def test_add_documents(self, vectorstore: VectorStore) -> None:
        """Test adding documents into the vectorstore."""
        original_documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(original_documents)
        documents = vectorstore.similarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
        ]
        # Verify that the original document object does not get mutated!
        # (e.g., an ID is added to the original document object)
        assert original_documents == [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

    def test_vectorstore_still_empty(self, vectorstore: VectorStore) -> None:
        """This test should follow a test that adds documents.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        assert vectorstore.similarity_search("foo", k=1) == []

    def test_deleting_documents(self, vectorstore: VectorStore) -> None:
        """Test deleting documents from the vectorstore."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents, ids=["1", "2"])
        assert ids == ["1", "2"]
        vectorstore.delete(["1"])
        documents = vectorstore.similarity_search("foo", k=1)
        assert documents == [Document(page_content="bar", metadata={"id": 2}, id="2")]

    def test_deleting_bulk_documents(self, vectorstore: VectorStore) -> None:
        """Test that we can delete several documents at once."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
            Document(page_content="baz", metadata={"id": 3}),
        ]

        vectorstore.add_documents(documents, ids=["1", "2", "3"])
        vectorstore.delete(["1", "2"])
        documents = vectorstore.similarity_search("foo", k=1)
        assert documents == [Document(page_content="baz", metadata={"id": 3}, id="3")]

    def test_delete_missing_content(self, vectorstore: VectorStore) -> None:
        """Deleting missing content should not raise an exception."""
        vectorstore.delete(["1"])
        vectorstore.delete(["1", "2", "3"])

    def test_add_documents_with_ids_is_idempotent(
        self, vectorstore: VectorStore
    ) -> None:
        """Adding by ID should be idempotent."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        vectorstore.add_documents(documents, ids=["1", "2"])
        vectorstore.add_documents(documents, ids=["1", "2"])
        documents = vectorstore.similarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id="2"),
            Document(page_content="foo", metadata={"id": 1}, id="1"),
        ]

    def test_add_documents_by_id_with_mutation(self, vectorstore: VectorStore) -> None:
        """Test that we can overwrite by ID using add_documents."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

        vectorstore.add_documents(documents=documents, ids=["1", "2"])

        # Now over-write content of ID 1
        new_documents = [
            Document(
                page_content="new foo", metadata={"id": 1, "some_other_field": "foo"}
            ),
        ]

        vectorstore.add_documents(documents=new_documents, ids=["1"])

        # Check that the content has been updated
        documents = vectorstore.similarity_search("new foo", k=2)
        assert documents == [
            Document(
                id="1",
                page_content="new foo",
                metadata={"id": 1, "some_other_field": "foo"},
            ),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]

    def test_get_by_ids(self, vectorstore: VectorStore) -> None:
        """Test get by IDs."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents, ids=["1", "2"])
        retrieved_documents = vectorstore.get_by_ids(ids)
        assert retrieved_documents == [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

    def test_get_by_ids_missing(self, vectorstore: VectorStore) -> None:
        """Test get by IDs with missing IDs."""
        # This should not raise an exception
        documents = vectorstore.get_by_ids(["1", "2", "3"])
        assert documents == []

    def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        """Run add_documents tests."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents)
        assert vectorstore.get_by_ids(ids) == [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        """Test that add_documentsing with existing IDs is idempotent."""
        documents = [
            Document(id="foo", page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents)
        assert "foo" in ids
        assert vectorstore.get_by_ids(ids) == [
            Document(page_content="foo", metadata={"id": 1}, id="foo"),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]


class AsyncReadWriteTestSuite(BaseStandardTests):
    """Test suite for checking the **async** read-write API of a vectorstore.

    This test suite verifies the basic read-write API of a vectorstore.

    The test suite is designed for asynchronous vectorstores.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty vectorstore for each test.

    The fixture should use the `get_embeddings` method to get a pre-defined
    embeddings model that should be used for this test suite.
    """

    @abstractmethod
    @pytest.fixture
    async def vectorstore(self) -> VectorStore:
        """Get the vectorstore class to test.

        The returned vectorstore should be EMPTY.
        """

    @staticmethod
    def get_embeddings() -> Embeddings:
        """A pre-defined embeddings model that should be used for this test."""
        return DeterministicFakeEmbedding(
            size=EMBEDDING_SIZE,
        )

    async def test_vectorstore_is_empty(self, vectorstore: VectorStore) -> None:
        """Test that the vectorstore is empty."""
        assert await vectorstore.asimilarity_search("foo", k=1) == []

    async def test_add_documents(self, vectorstore: VectorStore) -> None:
        """Test adding documents into the vectorstore."""
        original_documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(original_documents)
        documents = await vectorstore.asimilarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
        ]

        # Verify that the original document object does not get mutated!
        # (e.g., an ID is added to the original document object)
        assert original_documents == [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

    async def test_vectorstore_still_empty(self, vectorstore: VectorStore) -> None:
        """This test should follow a test that adds documents.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        assert await vectorstore.asimilarity_search("foo", k=1) == []

    async def test_deleting_documents(self, vectorstore: VectorStore) -> None:
        """Test deleting documents from the vectorstore."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents, ids=["1", "2"])
        assert ids == ["1", "2"]
        await vectorstore.adelete(["1"])
        documents = await vectorstore.asimilarity_search("foo", k=1)
        assert documents == [Document(page_content="bar", metadata={"id": 2}, id="2")]

    async def test_deleting_bulk_documents(self, vectorstore: VectorStore) -> None:
        """Test that we can delete several documents at once."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
            Document(page_content="baz", metadata={"id": 3}),
        ]

        await vectorstore.aadd_documents(documents, ids=["1", "2", "3"])
        await vectorstore.adelete(["1", "2"])
        documents = await vectorstore.asimilarity_search("foo", k=1)
        assert documents == [Document(page_content="baz", metadata={"id": 3}, id="3")]

    async def test_delete_missing_content(self, vectorstore: VectorStore) -> None:
        """Deleting missing content should not raise an exception."""
        await vectorstore.adelete(["1"])
        await vectorstore.adelete(["1", "2", "3"])

    async def test_add_documents_with_ids_is_idempotent(
        self, vectorstore: VectorStore
    ) -> None:
        """Adding by ID should be idempotent."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        await vectorstore.aadd_documents(documents, ids=["1", "2"])
        await vectorstore.aadd_documents(documents, ids=["1", "2"])
        documents = await vectorstore.asimilarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id="2"),
            Document(page_content="foo", metadata={"id": 1}, id="1"),
        ]

    async def test_add_documents_by_id_with_mutation(
        self, vectorstore: VectorStore
    ) -> None:
        """Test that we can overwrite by ID using add_documents."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]

        await vectorstore.aadd_documents(documents=documents, ids=["1", "2"])

        # Now over-write content of ID 1
        new_documents = [
            Document(
                page_content="new foo", metadata={"id": 1, "some_other_field": "foo"}
            ),
        ]

        await vectorstore.aadd_documents(documents=new_documents, ids=["1"])

        # Check that the content has been updated
        documents = await vectorstore.asimilarity_search("new foo", k=2)
        assert documents == [
            Document(
                id="1",
                page_content="new foo",
                metadata={"id": 1, "some_other_field": "foo"},
            ),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]

    async def test_get_by_ids(self, vectorstore: VectorStore) -> None:
        """Test get by IDs."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents, ids=["1", "2"])
        retrieved_documents = await vectorstore.aget_by_ids(ids)
        assert retrieved_documents == [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

    async def test_get_by_ids_missing(self, vectorstore: VectorStore) -> None:
        """Test get by IDs with missing IDs."""
        # This should not raise an exception
        assert await vectorstore.aget_by_ids(["1", "2", "3"]) == []

    async def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        """Run add_documents tests."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents)
        assert await vectorstore.aget_by_ids(ids) == [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

    async def test_add_documents_with_existing_ids(
        self, vectorstore: VectorStore
    ) -> None:
        """Test that add_documentsing with existing IDs is idempotent."""
        documents = [
            Document(id="foo", page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents)
        assert "foo" in ids
        assert await vectorstore.aget_by_ids(ids) == [
            Document(page_content="foo", metadata={"id": 1}, id="foo"),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]
