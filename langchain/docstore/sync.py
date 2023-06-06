"""Module contains doc for syncing from docstore to vectorstores."""
from typing import TypedDict, Sequence

from langchain.docstore.base import ArtifactLayer, Selector
from langchain.vectorstores import VectorStore


class SyncResult(TypedDict):
    """Syncing result."""

    first_n_errors: Sequence[str]
    """First n errors that occurred during syncing."""
    num_added: int
    """Number of added documents."""
    num_updated: int
    """Number of updated documents because they were not up to date."""
    num_deleted: int
    """Number of deleted documents."""
    num_skipped: int
    """Number of skipped documents because they were already up to date."""


def sync(
    artifact_layer: ArtifactLayer, vector_store: VectorStore, selector: Selector
) -> SyncResult:
    """Sync the given artifact layer with the given vector store."""

    matching_documents = artifact_layer.get_matching_documents(selector)
    # IDs must fit into memory for this to work.
    upsert_info = vector_store.upsert_by_id(documents=matching_documents)
    # Non-intuitive interface, but simple to implement
    # (maybe we can have a better solution though)
    num_deleted = vector_store.delete_non_matching_ids(upsert_info["ids_in_request"])

    return {
        "first_n_errors": [],
        "num_added": upsert_info["num_added"],
        "num_updated": upsert_info["num_updated"],
        "num_skipped": upsert_info["num_skipped"],
        "num_deleted": num_deleted,
    }
