"""Test suite to check indexer implementations."""

import inspect
import uuid
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator

import pytest
from langchain_core.documents import Document
from langchain_core.indexing import AsyncDocumentIndexer, DocumentIndexer


class DocumentIndexerTestSuite(ABC):
    """Test suite for checking the read-write of a document indexer.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty indexer for each test.
    """

    @abstractmethod
    @pytest.fixture
    def indexer(self) -> Generator[DocumentIndexer, None, None]:
        """Get the indexer."""

    def test_upsert_documents_has_no_ids(self, indexer: DocumentIndexer) -> None:
        """Verify that there is not parameter called ids in upsert"""
        signature = inspect.signature(indexer.upsert)
        assert "ids" not in signature.parameters

    def test_upsert_no_ids(self, indexer: DocumentIndexer) -> None:
        """Upsert works with documents that do not have IDs.

        At the moment, the ID field in documents is optional.
        """
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = indexer.upsert(documents)
        ids = sorted(response["succeeded"])

        # Ordering is not guaranteed, need to test carefully
        documents = indexer.get(ids)
        sorted_documents = sorted(documents, key=lambda x: x.id)

        if sorted_documents[0].page_content == "bar":
            assert sorted_documents[0] == Document(
                page_content="bar", metadata={"id": 2}, id=ids[0]
            )
            assert sorted_documents[1] == Document(
                page_content="foo", metadata={"id": 1}, id=ids[1]
            )
        else:
            assert sorted_documents[0] == Document(
                page_content="foo", metadata={"id": 1}, id=ids[0]
            )
            assert sorted_documents[1] == Document(
                page_content="bar", metadata={"id": 2}, id=ids[1]
            )

    def test_upsert_some_ids(self, indexer: DocumentIndexer) -> None:
        """Test an upsert where some docs have ids and some dont."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = indexer.upsert(documents)
        ids = response["succeeded"]
        other_id = list(set(ids) - {foo_uuid})[0]
        assert response["failed"] == []
        assert foo_uuid in ids
        # Ordering is not guaranteed, so we use a set.
        documents = indexer.get(ids)
        first_doc = documents[0]
        if first_doc.id == foo_uuid:
            assert documents == [
                Document(page_content="foo", metadata={"id": 1}, id=foo_uuid),
                Document(page_content="bar", metadata={"id": 2}, id=other_id),
            ]
        else:
            assert documents == [
                Document(page_content="bar", metadata={"id": 2}, id=other_id),
                Document(page_content="foo", metadata={"id": 1}, id=foo_uuid),
            ]

    def test_upsert_overwrites(self, indexer: DocumentIndexer) -> None:
        """Test that upsert overwrites existing content."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"bar": 1}),
        ]
        response = indexer.upsert(documents)
        ids = response["succeeded"]
        assert response["failed"] == []

        assert indexer.get(ids) == [
            Document(page_content="foo", metadata={"bar": 1}, id=foo_uuid),
        ]

        # Now let's overwrite foo
        indexer.upsert(
            [Document(id=foo_uuid, page_content="foo2", metadata={"meow": 2})]
        )
        documents = indexer.get([foo_uuid])
        assert documents == [
            Document(page_content="foo2", metadata={"meow": 2}, id=foo_uuid)
        ]

    def test_delete_missing_docs(self, indexer: DocumentIndexer) -> None:
        """Verify that we can delete docs that aren't there."""
        assert indexer.get(["1"]) == []  # Should be empty.

        delete_response = indexer.delete(["1"])
        if "num_deleted" in delete_response:
            assert delete_response["num_deleted"] == 0

        if "num_failed" in delete_response:
            # Deleting a missing an ID is **not** failure!!
            assert delete_response["num_failed"] == 0

        if "succeeded" in delete_response:
            # There was nothing to delete!
            assert delete_response["succeeded"] == []

        if "failed" in delete_response:
            # Nothing should have failed
            assert delete_response["failed"] == []

    def test_delete_semantics(self, indexer: DocumentIndexer) -> None:
        """Test deletion of content has appropriate semantics."""
        # Let's index a document first.
        foo_uuid = str(uuid.UUID(int=7))
        upsert_response = indexer.upsert(
            [Document(id=foo_uuid, page_content="foo", metadata={})]
        )
        assert upsert_response == {"succeeded": [foo_uuid], "failed": []}

        delete_response = indexer.delete(["missing_id", foo_uuid])

        if "num_deleted" in delete_response:
            assert delete_response["num_deleted"] == 1

        if "num_failed" in delete_response:
            # Deleting a missing an ID is **not** failure!!
            assert delete_response["num_failed"] == 0

        if "succeeded" in delete_response:
            # There was nothing to delete!
            assert delete_response["succeeded"] == [foo_uuid]

        if "failed" in delete_response:
            # Nothing should have failed
            assert delete_response["failed"] == []

    def test_bulk_delete(self, indexer: DocumentIndexer) -> None:
        """Test that we can delete several documents at once."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
            Document(id="3", page_content="baz", metadata={"id": 3}),
        ]

        indexer.upsert(documents)
        indexer.delete(["1", "2"])
        assert indexer.get(["1", "2", "3"]) == [
            Document(page_content="baz", metadata={"id": 3}, id="3")
        ]

    def test_delete_no_args(self, indexer: DocumentIndexer) -> None:
        """Test delete with no args raises ValueError."""
        with pytest.raises(ValueError):
            indexer.delete()

    def test_delete_missing_content(self, indexer: DocumentIndexer) -> None:
        """Deleting missing content should not raise an exception."""
        indexer.delete(["1"])
        indexer.delete(["1", "2", "3"])

    def test_get_with_missing_ids(self, indexer: DocumentIndexer) -> None:
        """Test get with missing IDs."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]
        upsert_response = indexer.upsert(documents)
        assert upsert_response == {
            "succeeded": ["1", "2"],
            "failed": [],
        }
        retrieved_documents = indexer.get(["1", "2", "3", "4"])
        # The ordering is not guaranteed, so we use a set.
        assert sorted(retrieved_documents, key=lambda x: x.id) == [
            Document(page_content="foo", metadata={"id": 1}, id="1"),
            Document(page_content="bar", metadata={"id": 2}, id="2"),
        ]

    def test_get_missing(self, indexer: DocumentIndexer) -> None:
        """Test get by IDs with missing IDs."""
        # This should not raise an exception
        documents = indexer.get(["1", "2", "3"])
        assert documents == []


class AsyncDocumentIndexerTestSuite(ABC):
    """Test suite for checking the read-write of a document indexer.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty indexer for each test.
    """

    @abstractmethod
    @pytest.fixture
    async def indexer(self) -> AsyncGenerator[AsyncDocumentIndexer, None]:
        """Get the indexer."""

    async def test_upsert_documents_has_no_ids(self, indexer: DocumentIndexer) -> None:
        """Verify that there is not parameter called ids in upsert"""
        signature = inspect.signature(indexer.upsert)
        assert "ids" not in signature.parameters

    async def test_upsert_no_ids(self, indexer: DocumentIndexer) -> None:
        """Upsert works with documents that do not have IDs.

        At the moment, the ID field in documents is optional.
        """
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = await indexer.upsert(documents)
        ids = sorted(response["succeeded"])

        # Ordering is not guaranteed, need to test carefully
        documents = await indexer.get(ids)
        sorted_documents = sorted(documents, key=lambda x: x.id)

        if sorted_documents[0].page_content == "bar":
            assert sorted_documents[0] == Document(
                page_content="bar", metadata={"id": 2}, id=ids[0]
            )
            assert sorted_documents[1] == Document(
                page_content="foo", metadata={"id": 1}, id=ids[1]
            )
        else:
            assert sorted_documents[0] == Document(
                page_content="foo", metadata={"id": 1}, id=ids[0]
            )
            assert sorted_documents[1] == Document(
                page_content="bar", metadata={"id": 2}, id=ids[1]
            )

    async def test_upsert_some_ids(self, indexer: DocumentIndexer) -> None:
        """Test an upsert where some docs have ids and some dont."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = await indexer.upsert(documents)
        ids = response["succeeded"]
        other_id = list(set(ids) - {foo_uuid})[0]
        assert response["failed"] == []
        assert foo_uuid in ids
        # Ordering is not guaranteed, so we use a set.
        documents = await indexer.get(ids)
        first_doc = documents[0]
        if first_doc.id == foo_uuid:
            assert documents == [
                Document(page_content="foo", metadata={"id": 1}, id=foo_uuid),
                Document(page_content="bar", metadata={"id": 2}, id=other_id),
            ]
        else:
            assert documents == [
                Document(page_content="bar", metadata={"id": 2}, id=other_id),
                Document(page_content="foo", metadata={"id": 1}, id=foo_uuid),
            ]

    async def test_upsert_overwrites(self, indexer: DocumentIndexer) -> None:
        """Test that upsert overwrites existing content."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"bar": 1}),
        ]
        response = await indexer.upsert(documents)
        ids = response["succeeded"]
        assert response["failed"] == []

        assert await indexer.get(ids) == [
            Document(page_content="foo", metadata={"bar": 1}, id=foo_uuid),
        ]

        # Now let's overwrite foo
        await indexer.upsert(
            [Document(id=foo_uuid, page_content="foo2", metadata={"meow": 2})]
        )
        documents = await indexer.get([foo_uuid])
        assert documents == [
            Document(page_content="foo2", metadata={"meow": 2}, id=foo_uuid)
        ]

    async def test_delete_missing_docs(self, indexer: DocumentIndexer) -> None:
        """Verify that we can delete docs that aren't there."""
        assert await indexer.get(["1"]) == []  # Should be empty.

        delete_response = await indexer.delete(["1"])
        if "num_deleted" in delete_response:
            assert delete_response["num_deleted"] == 0

        if "num_failed" in delete_response:
            # Deleting a missing an ID is **not** failure!!
            assert delete_response["num_failed"] == 0

        if "succeeded" in delete_response:
            # There was nothing to delete!
            assert delete_response["succeeded"] == []

        if "failed" in delete_response:
            # Nothing should have failed
            assert delete_response["failed"] == []

    async def test_delete_semantics(self, indexer: DocumentIndexer) -> None:
        """Test deletion of content has appropriate semantics."""
        # Let's index a document first.
        foo_uuid = str(uuid.UUID(int=7))
        upsert_response = await indexer.upsert(
            [Document(id=foo_uuid, page_content="foo", metadata={})]
        )
        assert upsert_response == {"succeeded": [foo_uuid], "failed": []}

        delete_response = await indexer.delete(["missing_id", foo_uuid])

        if "num_deleted" in delete_response:
            assert delete_response["num_deleted"] == 1

        if "num_failed" in delete_response:
            # Deleting a missing an ID is **not** failure!!
            assert delete_response["num_failed"] == 0

        if "succeeded" in delete_response:
            # There was nothing to delete!
            assert delete_response["succeeded"] == [foo_uuid]

        if "failed" in delete_response:
            # Nothing should have failed
            assert delete_response["failed"] == []

    async def test_bulk_delete(self, indexer: DocumentIndexer) -> None:
        """Test that we can delete several documents at once."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
            Document(id="3", page_content="baz", metadata={"id": 3}),
        ]

        await indexer.upsert(documents)
        await indexer.delete(["1", "2"])
        assert await indexer.get(["1", "2", "3"]) == [
            Document(page_content="baz", metadata={"id": 3}, id="3")
        ]

    async def test_delete_no_args(self, indexer: DocumentIndexer) -> None:
        """Test delete with no args raises ValueError."""
        with pytest.raises(ValueError):
            await indexer.delete()

    async def test_delete_missing_content(self, indexer: DocumentIndexer) -> None:
        """Deleting missing content should not raise an exception."""
        await indexer.delete(["1"])
        await indexer.delete(["1", "2", "3"])

    async def test_get_with_missing_ids(self, indexer: DocumentIndexer) -> None:
        """Test get with missing IDs."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]
        upsert_response = await indexer.upsert(documents)
        assert upsert_response == {
            "succeeded": ["1", "2"],
            "failed": [],
        }
        retrieved_documents = await indexer.get(["1", "2", "3", "4"])
        # The ordering is not guaranteed, so we use a set.
        assert sorted(retrieved_documents, key=lambda x: x.id) == [
            Document(page_content="foo", metadata={"id": 1}, id="1"),
            Document(page_content="bar", metadata={"id": 2}, id="2"),
        ]

    async def test_get_missing(self, indexer: DocumentIndexer) -> None:
        """Test get by IDs with missing IDs."""
        # This should not raise an exception
        documents = await indexer.get(["1", "2", "3"])
        assert documents == []
