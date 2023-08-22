"""Module contains logic for indexing documents into vector stores."""
from __future__ import annotations

import hashlib
import json
import uuid
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from langchain.document_loaders.base import BaseLoader
from langchain.indexing.base import NAMESPACE_UUID, RecordManager
from langchain.pydantic_v1 import root_validator
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

T = TypeVar("T")


def _hash_string_to_uuid(input_string: str) -> uuid.UUID:
    """Hashes a string and returns the corresponding UUID."""
    hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


def _hash_nested_dict_to_uuid(data: dict) -> uuid.UUID:
    """Hashes a nested dictionary and returns the corresponding UUID."""
    serialized_data = json.dumps(data, sort_keys=True)
    hash_value = hashlib.sha1(serialized_data.encode("utf-8")).hexdigest()
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


class _HashedDocument(Document):
    """A hashed document with a unique ID."""

    uid: str
    hash_: str
    """The hash of the document including content and metadat."""
    content_hash: str
    """The hash of the document content."""
    metadata_hash: str
    """The hash of the document metadata."""

    @root_validator(pre=True)
    def calculate_hashes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Root validator to calculate content and metadata hash."""
        content = values.get("page_content", "")
        metadata = values.get("metadata", {})

        content_hash = str(_hash_string_to_uuid(content))
        try:
            metadata_hash = str(_hash_nested_dict_to_uuid(metadata))
        except Exception as e:
            raise ValueError(
                f"Failed to hash metadata: {e}. "
                f"Please use a dict that can be serialized using json."
            )

        values["content_hash"] = content_hash
        values["metadata_hash"] = metadata_hash
        values["hash_"] = str(_hash_string_to_uuid(content_hash + metadata_hash))

        _uid = values.get("uid", None)

        if _uid is None:
            values["uid"] = values["hash_"]
        return values

    def to_document(self) -> Document:
        """Return a Document object."""
        return Document(
            page_content=self.page_content,
            metadata=self.metadata,
        )

    @classmethod
    def from_document(
        cls, document: Document, *, uid: Optional[str] = None
    ) -> _HashedDocument:
        """Create a HashedDocument from a Document."""
        return cls(
            uid=uid,
            page_content=document.page_content,
            metadata=document.metadata,
        )


def _batch(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    """Utility batching function."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def _get_source_id_assigner(
    source_id_key: Union[str, Callable[[Document], str], None],
) -> Callable[[Document], Union[str, None]]:
    """Get the source id from the document."""
    if source_id_key is None:
        return lambda doc: None
    elif isinstance(source_id_key, str):
        return lambda doc: doc.metadata[source_id_key]
    elif callable(source_id_key):
        return source_id_key
    else:
        raise ValueError(
            f"source_id_key should be either None, a string or a callable. "
            f"Got {source_id_key} of type {type(source_id_key)}."
        )


# PUBLIC API


class IndexingResult(TypedDict):
    """Return a detailed a breakdown of the result of the indexing operation."""

    num_added: Optional[int]
    """Number of added documents."""
    num_updated: Optional[int]
    """Number of updated documents because they were not up to date."""
    num_deleted: Optional[int]
    """Number of deleted documents."""
    num_skipped: Optional[int]
    """Number of skipped documents because they were already up to date."""


def index(
    loader: BaseLoader,
    record_manager: RecordManager,
    vector_store: VectorStore,
    *,
    batch_size: int = 100,
    delete_mode: Literal["incremental", "full", None] = None,
    source_id_key: Union[str, Callable[[Document], str], None] = None,
) -> IndexingResult:
    """Index data from the loader into the vector store.

    Indexing functionality uses a manager to keep track of which documents
    are in the vector store.

    This allows us to keep track of which documents were updated, and which
    documents were deleted, which documents should be skipped.

    For the time being, documents are indexed using their hashes, and users
     are not able to specify the uid of the document.

    IMPORTANT:
       if auto_cleanup is set to True, the loader should be returning
       the entire dataset, and not just a subset of the dataset.
       Otherwise, the auto_cleanup will remove documents that it is not
       supposed to.

    Args:
        loader: Data loader that returns a sequence of documents.
        record_manager: Timestamped set to keep track of which documents were
                         updated.
        vector_store: Vector store to index the documents into.
        batch_size: Batch size to use when indexing.
        delete_mode: How to handle clean up of documents.
            - Incremental: Cleans up all documents that haven't been updated AND
                           that are associated with source ids that were seen
                           during indexing.
                           Clean up is done continuously during indexing helping
                           to minimize the probability of users seeing duplicated
                           content.
            - Full: Delete all documents that haven to been returned by the loader.
                    Clean up runs after all documents have been indexed.
                    This means that users may see duplicated content during indexing.
            - None: Do not delete any documents.
        source_id_key: Optional key that helps identify the original source
            of the document.

    Returns:
        Indexing result which contains information about how many documents
        were added, updated, deleted, or skipped.
    """
    if delete_mode == "incremental" and source_id_key is None:
        raise ValueError("Source id key is required when delete mode is incremental.")

    # Check that the Vectorstore has required methods implemented
    methods = ["delete", "add_documents"]

    for method in methods:
        if not hasattr(vector_store, method):
            raise NotImplementedError(
                f"Vectorstore {vector_store} does not have required method {method}"
            )

    try:
        doc_iterator = loader.lazy_load()
    except NotImplementedError:
        doc_iterator = iter(loader.load())

    source_id_assigner = _get_source_id_assigner(source_id_key)

    # Mark when the update started.
    index_start_dt = record_manager.get_time()
    num_added = 0
    num_skipped = 0
    num_updated = 0
    num_deleted = 0

    for doc_batch in _batch(batch_size, doc_iterator):
        hashed_docs = [
            _HashedDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in doc_batch
        ]

        source_ids: List[Optional[str]] = [
            source_id_assigner(doc.to_document()) for doc in hashed_docs
        ]

        if delete_mode == "incremental":
            # If the delete mode is incremental, source ids are required.
            for source_id, hashed_doc in zip(source_ids, hashed_docs):
                if source_id is None:
                    raise ValueError(
                        "Source ids are required when delete mode is incremental. "
                        f"Document that starts with "
                        f"content: {hashed_doc.page_content[:100]} was not assigned "
                        f"as source id."
                    )
            source_ids = cast(List[str], source_ids)

        exists_batch = record_manager.exists([doc.uid for doc in hashed_docs])

        # Filter out documents that already exist in the record store.
        uids = []
        docs_to_index = []
        for doc, smart_doc, doc_exists in zip(doc_batch, hashed_docs, exists_batch):
            if doc_exists:
                # Must be updated to refresh timestamp.
                record_manager.update([smart_doc.uid], time_at_least=index_start_dt)
                num_skipped += 1
                continue
            uids.append(smart_doc.uid)
            docs_to_index.append(doc)

        # Be pessimistic and assume that all vector store write will fail.
        # First write to vector store
        if docs_to_index:
            vector_store.add_documents(docs_to_index, ids=uids)
            num_added += len(docs_to_index)

        # And only then update the record store.
        # Update ALL records, even if they already exist since we want to refresh
        # their timestamp.
        record_manager.update(
            [doc.uid for doc in hashed_docs],
            group_ids=source_ids,
            time_at_least=index_start_dt,
        )

        # If source IDs are provided, we can do the deletion incrementally!
        if delete_mode == "incremental":
            # Get the uids of the documents that were not returned by the loader.
            uids_to_delete = record_manager.list_keys(
                group_ids=source_ids, before=index_start_dt
            )
            if uids_to_delete:
                # Then delete from vector store.
                vector_store.delete(uids_to_delete)
                # First delete from record store.
                record_manager.delete_keys(uids_to_delete)
                num_deleted += len(uids_to_delete)

    if delete_mode == "full":
        uids_to_delete = record_manager.list_keys(before=index_start_dt)

        if uids_to_delete:
            # Then delete from vector store.
            vector_store.delete(uids_to_delete)
            # First delete from record store.
            record_manager.delete_keys(uids_to_delete)
            num_deleted = len(uids_to_delete)

    return {
        "num_added": num_added,
        "num_updated": num_updated,
        "num_skipped": num_skipped,
        "num_deleted": num_deleted,
    }
