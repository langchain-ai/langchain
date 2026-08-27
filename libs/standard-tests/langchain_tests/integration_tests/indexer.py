"""Test suite to check index implementations.

Standard tests for the `DocumentIndex` abstraction

We don't recommend implementing externally managed `DocumentIndex` abstractions at this
time.
"""

import inspect
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator

import pytest
from langchain_core.documents import Document
from langchain_core.indexing.base import DocumentIndex


class DocumentIndexerTestSuite(ABC):
    """Test suite for checking the read-write of a document index.

    Implementers should subclass this test suite and provide a fixture that returns an
    empty index for each test.
    """

    @abstractmethod
    @pytest.fixture
    def index(self) -> Generator[DocumentIndex, None, None]:
        """Get the index."""

    def test_upsert_documents_has_no_ids(self, index: DocumentIndex) -> None:
        """Verify that there is no parameter called IDs in upsert."""
        signature = inspect.signature(index.upsert)
        assert "ids" not in signature.parameters

    def test_upsert_no_ids(self, index: DocumentIndex) -> None:
        """Upsert works with documents that do not have IDs.

        At the moment, the ID field in documents is optional.
        """
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = index.upsert(documents)
        ids = sorted(response["succeeded"])

        # Ordering is not guaranteed, need to test carefully
        documents = index.get(ids)
        sorted_documents = sorted(documents, key=lambda x: x.id or "")

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

    def test_upsert_some_ids(self, index: DocumentIndex) -> None:
        """Test an upsert where some docs have IDs and some don't."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = index.upsert(documents)
        ids = response["succeeded"]
        other_id = next(iter(set(ids) - {foo_uuid}))
        assert response["failed"] == []
        assert foo_uuid in ids
        # Ordering is not guaranteed, so we use a set.
        documents = index.get(ids)
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

    def test_upsert_overwrites(self, index: DocumentIndex) -> None:
        """Test that upsert overwrites existing content."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"bar": 1}),
        ]
        response = index.upsert(documents)
        ids = response["succeeded"]
        assert response["failed"] == []

        assert index.get(ids) == [
            Document(page_content="foo", metadata={"bar": 1}, id=foo_uuid),
        ]

        # Now let's overwrite foo
        index.upsert([Document(id=foo_uuid, page_content="foo2", metadata={"meow": 2})])
        documents = index.get([foo_uuid])
        assert documents == [
            Document(page_content="foo2", metadata={"meow": 2}, id=foo_uuid)
        ]

    def test_delete_missing_docs(self, index: DocumentIndex) -> None:
        """Verify that we can delete docs that aren't there."""
        assert index.get(["1"]) == []  # Should be empty.

        delete_response = index.delete(["1"])
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

    def test_delete_semantics(self, index: DocumentIndex) -> None:
        """Test deletion of content has appropriate semantics."""
        # Let's index a document first.
        foo_uuid = str(uuid.UUID(int=7))
        upsert_response = index.upsert(
            [Document(id=foo_uuid, page_content="foo", metadata={})]
        )
        assert upsert_response == {"succeeded": [foo_uuid], "failed": []}

        delete_response = index.delete(["missing_id", foo_uuid])

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

    def test_bulk_delete(self, index: DocumentIndex) -> None:
        """Test that we can delete several documents at once."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
            Document(id="3", page_content="baz", metadata={"id": 3}),
        ]

        index.upsert(documents)
        index.delete(["1", "2"])
        assert index.get(["1", "2", "3"]) == [
            Document(page_content="baz", metadata={"id": 3}, id="3")
        ]

    def test_delete_no_args(self, index: DocumentIndex) -> None:
        """Test delete with no args raises `ValueError`."""
        with pytest.raises(ValueError):  # noqa: PT011
            index.delete()

    def test_delete_missing_content(self, index: DocumentIndex) -> None:
        """Deleting missing content should not raise an exception."""
        index.delete(["1"])
        index.delete(["1", "2", "3"])

    def test_get_with_missing_ids(self, index: DocumentIndex) -> None:
        """Test get with missing IDs."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]
        upsert_response = index.upsert(documents)
        assert upsert_response == {
            "succeeded": ["1", "2"],
            "failed": [],
        }
        retrieved_documents = index.get(["1", "2", "3", "4"])
        # The ordering is not guaranteed, so we use a set.
        assert sorted(retrieved_documents, key=lambda x: x.id or "") == [
            Document(page_content="foo", metadata={"id": 1}, id="1"),
            Document(page_content="bar", metadata={"id": 2}, id="2"),
        ]

    def test_get_missing(self, index: DocumentIndex) -> None:
        """Test get by IDs with missing IDs."""
        # This should not raise an exception
        documents = index.get(["1", "2", "3"])
        assert documents == []


class AsyncDocumentIndexTestSuite(ABC):
    """Test suite for checking the read-write of a document index.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty index for each test.
    """

    @abstractmethod
    @pytest.fixture
    async def index(self) -> AsyncGenerator[DocumentIndex, None]:
        """Get the index."""

    async def test_upsert_documents_has_no_ids(self, index: DocumentIndex) -> None:
        """Verify that there is not parameter called IDs in upsert."""
        signature = inspect.signature(index.upsert)
        assert "ids" not in signature.parameters

    async def test_upsert_no_ids(self, index: DocumentIndex) -> None:
        """Upsert works with documents that do not have IDs.

        At the moment, the ID field in documents is optional.
        """
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = await index.aupsert(documents)
        ids = sorted(response["succeeded"])

        # Ordering is not guaranteed, need to test carefully
        documents = await index.aget(ids)
        sorted_documents = sorted(documents, key=lambda x: x.id or "")

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

    async def test_upsert_some_ids(self, index: DocumentIndex) -> None:
        """Test an upsert where some docs have IDs and some don't."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        response = await index.aupsert(documents)
        ids = response["succeeded"]
        other_id = next(iter(set(ids) - {foo_uuid}))
        assert response["failed"] == []
        assert foo_uuid in ids
        # Ordering is not guaranteed, so we use a set.
        documents = await index.aget(ids)
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

    async def test_upsert_overwrites(self, index: DocumentIndex) -> None:
        """Test that upsert overwrites existing content."""
        foo_uuid = str(uuid.UUID(int=7))
        documents = [
            Document(id=foo_uuid, page_content="foo", metadata={"bar": 1}),
        ]
        response = await index.aupsert(documents)
        ids = response["succeeded"]
        assert response["failed"] == []

        assert await index.aget(ids) == [
            Document(page_content="foo", metadata={"bar": 1}, id=foo_uuid),
        ]

        # Now let's overwrite foo
        await index.aupsert(
            [Document(id=foo_uuid, page_content="foo2", metadata={"meow": 2})]
        )
        documents = await index.aget([foo_uuid])
        assert documents == [
            Document(page_content="foo2", metadata={"meow": 2}, id=foo_uuid)
        ]

    async def test_delete_missing_docs(self, index: DocumentIndex) -> None:
        """Verify that we can delete docs that aren't there."""
        assert await index.aget(["1"]) == []  # Should be empty.

        delete_response = await index.adelete(["1"])
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

    async def test_delete_semantics(self, index: DocumentIndex) -> None:
        """Test deletion of content has appropriate semantics."""
        # Let's index a document first.
        foo_uuid = str(uuid.UUID(int=7))
        upsert_response = await index.aupsert(
            [Document(id=foo_uuid, page_content="foo", metadata={})]
        )
        assert upsert_response == {"succeeded": [foo_uuid], "failed": []}

        delete_response = await index.adelete(["missing_id", foo_uuid])

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

    async def test_bulk_delete(self, index: DocumentIndex) -> None:
        """Test that we can delete several documents at once."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
            Document(id="3", page_content="baz", metadata={"id": 3}),
        ]

        await index.aupsert(documents)
        await index.adelete(["1", "2"])
        assert await index.aget(["1", "2", "3"]) == [
            Document(page_content="baz", metadata={"id": 3}, id="3")
        ]

    async def test_delete_no_args(self, index: DocumentIndex) -> None:
        """Test delete with no args raises `ValueError`."""
        with pytest.raises(ValueError):  # noqa: PT011
            await index.adelete()

    async def test_delete_missing_content(self, index: DocumentIndex) -> None:
        """Deleting missing content should not raise an exception."""
        await index.adelete(["1"])
        await index.adelete(["1", "2", "3"])

    async def test_get_with_missing_ids(self, index: DocumentIndex) -> None:
        """Test get with missing IDs."""
        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(id="2", page_content="bar", metadata={"id": 2}),
        ]
        upsert_response = await index.aupsert(documents)
        assert upsert_response == {
            "succeeded": ["1", "2"],
            "failed": [],
        }
        retrieved_documents = await index.aget(["1", "2", "3", "4"])
        # The ordering is not guaranteed, so we use a set.
        assert sorted(retrieved_documents, key=lambda x: x.id or "") == [
            Document(page_content="foo", metadata={"id": 1}, id="1"),
            Document(page_content="bar", metadata={"id": 2}, id="2"),
        ]

    async def test_get_missing(self, index: DocumentIndex) -> None:
        """Test get by IDs with missing IDs."""
        # This should not raise an exception
        documents = await index.aget(["1", "2", "3"])
        assert documents == []
