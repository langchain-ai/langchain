"""Module contains doc for syncing from docstore to vectorstores."""
from __future__ import annotations

from itertools import islice
from typing import TypedDict, Sequence, Optional, TypeVar, Iterable, Iterator, List
from langchain.docstore.base import ArtifactStore, Selector
from langchain.vectorstores import VectorStore


class SyncResult(TypedDict):
    """Syncing result."""

    first_n_errors: Sequence[str]
    """First n errors that occurred during syncing."""
    num_added: Optional[int]
    """Number of added documents."""
    num_updated: Optional[int]
    """Number of updated documents because they were not up to date."""
    num_deleted: Optional[int]
    """Number of deleted documents."""
    num_skipped: Optional[int]
    """Number of skipped documents because they were already up to date."""


T = TypeVar("T")


def _batch(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    """Utility batching function."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


# SYNC IMPLEMENTATION


def sync(
    artifact_store: ArtifactStore,
    vector_store: VectorStore,
    selector: Selector,
    *,
    batch_size: int = 1000,
) -> SyncResult:
    """Sync the given artifact layer with the given vector store."""
    document_uids = artifact_store.list_document_ids(selector)

    all_uids = []
    # IDs must fit into memory for this to work.
    for uid_batch in _batch(batch_size, document_uids):
        all_uids.extend(uid_batch)
        document_batch = list(artifact_store.list_documents(Selector(uids=uid_batch)))
        upsert_info = vector_store.upsert_by_id(
            documents=document_batch, batch_size=batch_size
        )
    # Non-intuitive interface, but simple to implement
    # (maybe we can have a better solution though)
    num_deleted = vector_store.delete_non_matching_ids(all_uids)

    return {
        "first_n_errors": [],
        "num_added": None,
        "num_updated": None,
        "num_skipped": None,
        "num_deleted": None,
    }
