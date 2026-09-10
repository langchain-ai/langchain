"""Tests for metadata-only updates in indexing API."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import pytest_asyncio

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.indexing import InMemoryRecordManager, aindex, index
from langchain_core.indexing.in_memory import InMemoryDocumentIndex
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore


class MockLoader(BaseLoader):
    """A loader that returns a predefined list of documents."""

    def __init__(self, documents: list[Document]) -> None:
        """Initialize with documents."""
        self.documents = documents

    def lazy_load(self) -> Iterator[Document]:
        yield from self.documents

    async def alazy_load(self) -> AsyncIterator[Document]:
        for document in self.documents:
            yield document


@pytest.fixture
def record_manager() -> InMemoryRecordManager:
    """Fixture for InMemoryRecordManager."""
    rm = InMemoryRecordManager(namespace="test_meta")
    rm.create_schema()
    return rm


@pytest_asyncio.fixture
async def arecord_manager() -> InMemoryRecordManager:
    """Async fixture for InMemoryRecordManager."""
    rm = InMemoryRecordManager(namespace="test_meta_async")
    await rm.acreate_schema()
    return rm


class CountingEmbedding(DeterministicFakeEmbedding):
    """Embedding that counts calls to embed_documents."""

    embed_call_count: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_call_count += len(texts)
        return super().embed_documents(texts)


class UnsupportedVectorStore(VectorStore):
    """Vector store that implements delete and add_documents but not update_metadata."""

    def delete(
        self,
        ids: list[str] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> bool | None:
        return True

    def add_documents(
        self,
        documents: list[Document],  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str]:
        return []

    @classmethod
    def from_texts(
        cls,
        texts: list[str],  # noqa: ARG003
        embedding: Any,  # noqa: ARG003
        metadatas: list[dict] | None = None,  # noqa: ARG003
        **kwargs: Any,  # noqa: ARG003
    ) -> "UnsupportedVectorStore":
        return cls()

    def similarity_search(
        self,
        query: str,  # noqa: ARG002
        k: int = 4,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[Document]:
        return []


@pytest.fixture
def counting_vector_store() -> tuple[InMemoryVectorStore, CountingEmbedding]:
    """Vector store with an embedding instance that counts embedding calls."""
    embedder = CountingEmbedding(size=5)
    return InMemoryVectorStore(embedder), embedder


def test_in_memory_vector_store_update_metadata() -> None:
    """Test InMemoryVectorStore.update_metadata directly."""
    embeddings = DeterministicFakeEmbedding(size=5)
    vs = InMemoryVectorStore(embeddings)
    doc = Document(page_content="hello", metadata={"tag": "v1"})
    ids = vs.add_documents([doc], ids=["id1"])
    assert vs.store["id1"]["metadata"] == {"tag": "v1"}
    orig_vector = list(vs.store["id1"]["vector"])

    vs.update_metadata(ids, [{"tag": "v2", "new_field": True}])
    assert vs.store["id1"]["metadata"] == {"tag": "v2", "new_field": True}
    assert vs.store["id1"]["vector"] == orig_vector

    with pytest.raises(ValueError, match="ids and metadatas must have the same length"):
        vs.update_metadata(["id1", "id2"], [{"tag": "v3"}])


def test_in_memory_document_index_update_metadata() -> None:
    """Test InMemoryDocumentIndex.update_metadata directly."""
    idx = InMemoryDocumentIndex()
    doc = Document(id="id1", page_content="hello", metadata={"tag": "v1"})
    idx.upsert([doc])
    assert idx.store["id1"].metadata == {"tag": "v1"}

    idx.update_metadata(["id1"], [{"tag": "v2"}])
    assert idx.store["id1"].metadata == {"tag": "v2"}
    assert idx.store["id1"].page_content == "hello"

    with pytest.raises(ValueError, match="ids and metadatas must have the same length"):
        idx.update_metadata(["id1"], [{"a": 1}, {"b": 2}])


def test_index_metadata_update_avoid_reembedding(
    record_manager: InMemoryRecordManager,
    counting_vector_store: tuple[InMemoryVectorStore, CountingEmbedding],
) -> None:
    """Test that metadata_update=True updates metadata without re-embedding."""
    vs, embedder = counting_vector_store
    docs = [
        Document(page_content="Document 1", metadata={"source": "1", "version": 1}),
        Document(page_content="Document 2", metadata={"source": "2", "version": 1}),
    ]

    # First indexing: 2 documents added, embeddings calculated
    res1 = index(
        docs,
        record_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert res1 == {
        "num_added": 2,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 2

    # Second indexing: identical docs -> both skipped, 0 embeddings
    res2 = index(
        docs,
        record_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert res2 == {
        "num_added": 0,
        "num_updated": 0,
        "num_skipped": 2,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 2  # No new embeddings computed!

    # Third indexing: doc 1 metadata updated, doc 2 unchanged
    updated_docs = [
        Document(page_content="Document 1", metadata={"source": "1", "version": 2}),
        Document(page_content="Document 2", metadata={"source": "2", "version": 1}),
    ]
    res3 = index(
        updated_docs,
        record_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert res3 == {
        "num_added": 0,
        "num_updated": 1,
        "num_skipped": 1,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 2  # Still 0 new embeddings!

    # Verify vector store metadata was updated
    for entry in vs.store.values():
        if entry["text"] == "Document 1":
            assert entry["metadata"]["version"] == 2
        elif entry["text"] == "Document 2":
            assert entry["metadata"]["version"] == 1


def test_index_metadata_update_content_changed(
    record_manager: InMemoryRecordManager,
    counting_vector_store: tuple[InMemoryVectorStore, CountingEmbedding],
) -> None:
    """Test that content changes trigger normal re-embedding and deletion."""
    vs, embedder = counting_vector_store
    docs = [
        Document(page_content="Original content", metadata={"source": "1", "v": 1}),
    ]
    index(
        docs,
        record_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert embedder.embed_call_count == 1

    # Content changed -> must re-embed and delete old document
    changed_docs = [
        Document(page_content="New content", metadata={"source": "1", "v": 2}),
    ]
    res = index(
        changed_docs,
        record_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert res == {
        "num_added": 1,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 1,
    }
    assert embedder.embed_call_count == 2


def test_index_metadata_update_force_update(
    record_manager: InMemoryRecordManager,
    counting_vector_store: tuple[InMemoryVectorStore, CountingEmbedding],
) -> None:
    """Test that force_update=True forces re-embedding.

    Should re-embed even with metadata_update=True.
    """
    vs, embedder = counting_vector_store
    docs = [
        Document(page_content="Doc 1", metadata={"source": "1"}),
    ]
    index(
        docs,
        record_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert embedder.embed_call_count == 1

    res = index(
        docs,
        record_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        force_update=True,
        metadata_update=True,
    )
    assert res["num_updated"] == 1
    assert embedder.embed_call_count == 2


def test_index_metadata_update_with_full_cleanup(
    record_manager: InMemoryRecordManager,
    counting_vector_store: tuple[InMemoryVectorStore, CountingEmbedding],
) -> None:
    """Test metadata_update=True with cleanup='full'."""
    vs, embedder = counting_vector_store
    docs = [
        Document(page_content="Doc 1", metadata={"source": "1", "v": 1}),
        Document(page_content="Doc 2", metadata={"source": "2", "v": 1}),
    ]
    index(
        docs,
        record_manager,
        vs,
        cleanup="full",
        metadata_update=True,
    )

    # In second run, Doc 2 is removed, Doc 1 metadata updated
    run2_docs = [
        Document(page_content="Doc 1", metadata={"source": "1", "v": 2}),
    ]
    res = index(
        run2_docs,
        record_manager,
        vs,
        cleanup="full",
        metadata_update=True,
    )
    assert res == {
        "num_added": 0,
        "num_updated": 1,
        "num_skipped": 0,
        "num_deleted": 1,
    }
    assert embedder.embed_call_count == 2  # No new embeddings computed in run 2!
    assert len(vs.store) == 1


def test_index_metadata_update_document_index(
    record_manager: InMemoryRecordManager,
) -> None:
    """Test metadata_update=True with DocumentIndex."""
    doc_index = InMemoryDocumentIndex()
    docs = [
        Document(page_content="Test doc", metadata={"author": "Alice"}),
    ]
    res1 = index(
        docs,
        record_manager,
        doc_index,
        metadata_update=True,
    )
    assert res1["num_added"] == 1

    # Update metadata
    res2 = index(
        [Document(page_content="Test doc", metadata={"author": "Bob"})],
        record_manager,
        doc_index,
        metadata_update=True,
    )
    assert res2 == {
        "num_added": 0,
        "num_updated": 1,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    stored_docs = list(doc_index.store.values())
    assert stored_docs[0].metadata == {"author": "Bob"}


def test_index_metadata_update_unsupported_destination(
    record_manager: InMemoryRecordManager,
) -> None:
    """Test that a VectorStore without update_metadata raises ValueError."""
    unsupported_vs = UnsupportedVectorStore()
    with pytest.raises(
        ValueError, match="has not implemented the update_metadata method"
    ):
        index(
            [Document(page_content="hello", metadata={})],
            record_manager,
            unsupported_vs,
            metadata_update=True,
        )


@pytest.mark.asyncio
async def test_aindex_metadata_update_avoid_reembedding(
    arecord_manager: InMemoryRecordManager,
    counting_vector_store: tuple[InMemoryVectorStore, CountingEmbedding],
) -> None:
    """Test async aindex with metadata_update=True."""
    vs, embedder = counting_vector_store
    docs = [
        Document(page_content="Async doc 1", metadata={"source": "1", "tag": "a"}),
        Document(page_content="Async doc 2", metadata={"source": "2", "tag": "a"}),
    ]

    res1 = await aindex(
        docs,
        arecord_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert res1 == {
        "num_added": 2,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 2

    # Skip when unchanged
    res2 = await aindex(
        docs,
        arecord_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert res2 == {
        "num_added": 0,
        "num_updated": 0,
        "num_skipped": 2,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 2

    # Update metadata only
    updated_docs = [
        Document(page_content="Async doc 1", metadata={"source": "1", "tag": "b"}),
        Document(page_content="Async doc 2", metadata={"source": "2", "tag": "a"}),
    ]
    res3 = await aindex(
        updated_docs,
        arecord_manager,
        vs,
        cleanup="incremental",
        source_id_key="source",
        metadata_update=True,
    )
    assert res3 == {
        "num_added": 0,
        "num_updated": 1,
        "num_skipped": 1,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 2

    for entry in vs.store.values():
        if entry["text"] == "Async doc 1":
            assert entry["metadata"]["tag"] == "b"
        elif entry["text"] == "Async doc 2":
            assert entry["metadata"]["tag"] == "a"


def test_index_metadata_update_with_explicit_id(
    record_manager: InMemoryRecordManager,
    counting_vector_store: tuple[InMemoryVectorStore, CountingEmbedding],
) -> None:
    """Test metadata update when documents have explicit ids."""
    vs, embedder = counting_vector_store
    doc = Document(
        id="custom-1", page_content="Hello world", metadata={"status": "draft"}
    )

    # First indexing
    res1 = index([doc], record_manager, vs, metadata_update=True)
    assert res1 == {
        "num_added": 1,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 1
    assert vs.store["custom-1"]["metadata"] == {"status": "draft"}

    # Update metadata only with same id
    doc_updated = Document(
        id="custom-1", page_content="Hello world", metadata={"status": "published"}
    )
    res2 = index([doc_updated], record_manager, vs, metadata_update=True)
    assert res2 == {
        "num_added": 0,
        "num_updated": 1,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    assert embedder.embed_call_count == 1  # 0 new embeddings
    assert vs.store["custom-1"]["metadata"] == {"status": "published"}


def test_index_metadata_update_scoped_full(
    record_manager: InMemoryRecordManager,
    counting_vector_store: tuple[InMemoryVectorStore, CountingEmbedding],
) -> None:
    """Test metadata update with cleanup='scoped_full'."""
    vs, embedder = counting_vector_store
    docs = [
        Document(page_content="Doc 1", metadata={"source": "src1", "v": 1}),
        Document(page_content="Doc 2", metadata={"source": "src1", "v": 1}),
        Document(page_content="Doc 3", metadata={"source": "src2", "v": 1}),
    ]
    index(
        docs,
        record_manager,
        vs,
        cleanup="scoped_full",
        source_id_key="source",
        metadata_update=True,
    )
    assert embedder.embed_call_count == 3

    # Only provide src1 docs: Doc 1 updated metadata, Doc 2 omitted (deleted)
    # Doc 3 (src2) is NOT seen in this run, so scoped_full should NOT delete it!
    run2_docs = [
        Document(page_content="Doc 1", metadata={"source": "src1", "v": 2}),
    ]
    res = index(
        run2_docs,
        record_manager,
        vs,
        cleanup="scoped_full",
        source_id_key="source",
        metadata_update=True,
    )
    assert res == {
        "num_added": 0,
        "num_updated": 1,
        "num_skipped": 0,
        "num_deleted": 1,  # Doc 2 deleted
    }
    assert embedder.embed_call_count == 3  # No new embeddings computed in run 2!
    # Doc 1 and Doc 3 still in vector store
    assert len(vs.store) == 2


@pytest.mark.asyncio
async def test_aindex_metadata_update_unsupported_destination(
    arecord_manager: InMemoryRecordManager,
) -> None:
    """Test that a VectorStore without aupdate_metadata raises ValueError in aindex."""
    unsupported_vs = UnsupportedVectorStore()
    with pytest.raises(
        ValueError,
        match="has not implemented the aupdate_metadata or update_metadata method",
    ):
        await aindex(
            [Document(page_content="hello", metadata={})],
            arecord_manager,
            unsupported_vs,
            metadata_update=True,
        )
