"""Module contains logic for indexing documents into vector stores."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence
from itertools import islice
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from pydantic import model_validator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.exceptions import LangChainException
from langchain_core.indexing.base import DocumentIndex, RecordManager
from langchain_core.vectorstores import VectorStore

# Magic UUID to use as a namespace for hashing.
# Used to try and generate a unique UUID for each document
# from hashing the document content and metadata.
NAMESPACE_UUID = uuid.UUID(int=1984)


T = TypeVar("T")


def _hash_string_to_uuid(input_string: str) -> uuid.UUID:
    """Hashes a string and returns the corresponding UUID."""
    hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()  # noqa: S324
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


def _hash_nested_dict_to_uuid(data: dict[Any, Any]) -> uuid.UUID:
    """Hashes a nested dictionary and returns the corresponding UUID."""
    serialized_data = json.dumps(data, sort_keys=True)
    hash_value = hashlib.sha1(serialized_data.encode("utf-8")).hexdigest()  # noqa: S324
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


class _HashedDocument(Document):
    """A hashed document with a unique ID."""

    uid: str
    hash_: str
    """The hash of the document including content and metadata."""
    content_hash: str
    """The hash of the document content."""
    metadata_hash: str
    """The hash of the document metadata."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @model_validator(mode="before")
    @classmethod
    def calculate_hashes(cls, values: dict[str, Any]) -> Any:
        """Root validator to calculate content and metadata hash."""
        content = values.get("page_content", "")
        metadata = values.get("metadata", {})

        forbidden_keys = ("hash_", "content_hash", "metadata_hash")

        for key in forbidden_keys:
            if key in metadata:
                msg = (
                    f"Metadata cannot contain key {key} as it "
                    f"is reserved for internal use."
                )
                raise ValueError(msg)

        content_hash = str(_hash_string_to_uuid(content))

        try:
            metadata_hash = str(_hash_nested_dict_to_uuid(metadata))
        except Exception as e:
            msg = (
                f"Failed to hash metadata: {e}. "
                f"Please use a dict that can be serialized using json."
            )
            raise ValueError(msg) from e

        values["content_hash"] = content_hash
        values["metadata_hash"] = metadata_hash
        values["hash_"] = str(_hash_string_to_uuid(content_hash + metadata_hash))

        _uid = values.get("uid")

        if _uid is None:
            values["uid"] = values["hash_"]
        return values

    def to_document(self) -> Document:
        """Return a Document object."""
        return Document(
            id=self.uid,
            page_content=self.page_content,
            metadata=self.metadata,
        )

    @classmethod
    def from_document(
        cls, document: Document, *, uid: Optional[str] = None
    ) -> _HashedDocument:
        """Create a HashedDocument from a Document."""
        return cls(  # type: ignore[call-arg]
            uid=uid,  # type: ignore[arg-type]
            page_content=document.page_content,
            metadata=document.metadata,
        )


def _batch(size: int, iterable: Iterable[T]) -> Iterator[list[T]]:
    """Utility batching function."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


async def _abatch(size: int, iterable: AsyncIterable[T]) -> AsyncIterator[list[T]]:
    """Utility batching function."""
    batch: list[T] = []
    async for element in iterable:
        if len(batch) < size:
            batch.append(element)

        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


def _get_source_id_assigner(
    source_id_key: Union[str, Callable[[Document], str], None],
) -> Callable[[Document], Union[str, None]]:
    """Get the source id from the document."""
    if source_id_key is None:
        return lambda doc: None
    if isinstance(source_id_key, str):
        return lambda doc: doc.metadata[source_id_key]
    if callable(source_id_key):
        return source_id_key
    msg = (
        f"source_id_key should be either None, a string or a callable. "
        f"Got {source_id_key} of type {type(source_id_key)}."
    )
    raise ValueError(msg)


def _deduplicate_in_order(
    hashed_documents: Iterable[_HashedDocument],
) -> Iterator[_HashedDocument]:
    """Deduplicate a list of hashed documents while preserving order."""
    seen: set[str] = set()

    for hashed_doc in hashed_documents:
        if hashed_doc.hash_ not in seen:
            seen.add(hashed_doc.hash_)
            yield hashed_doc


class IndexingException(LangChainException):
    """Raised when an indexing operation fails."""


def _delete(
    vector_store: Union[VectorStore, DocumentIndex],
    ids: list[str],
) -> None:
    if isinstance(vector_store, VectorStore):
        delete_ok = vector_store.delete(ids)
        if delete_ok is not None and delete_ok is False:
            msg = "The delete operation to VectorStore failed."
            raise IndexingException(msg)
    elif isinstance(vector_store, DocumentIndex):
        delete_response = vector_store.delete(ids)
        if "num_failed" in delete_response and delete_response["num_failed"] > 0:
            msg = "The delete operation to DocumentIndex failed."
            raise IndexingException(msg)
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(vector_store)}."
        )
        raise TypeError(msg)


# PUBLIC API


class IndexingResult(TypedDict):
    """Return a detailed a breakdown of the result of the indexing operation."""

    num_added: int
    """Number of added documents."""
    num_updated: int
    """Number of updated documents because they were not up to date."""
    num_deleted: int
    """Number of deleted documents."""
    num_skipped: int
    """Number of skipped documents because they were already up to date."""


def index(
    docs_source: Union[BaseLoader, Iterable[Document]],
    record_manager: RecordManager,
    vector_store: Union[VectorStore, DocumentIndex],
    *,
    batch_size: int = 100,
    cleanup: Literal["incremental", "full", "scoped_full", None] = None,
    source_id_key: Union[str, Callable[[Document], str], None] = None,
    cleanup_batch_size: int = 1_000,
    force_update: bool = False,
    upsert_kwargs: Optional[dict[str, Any]] = None,
) -> IndexingResult:
    """Index data from the loader into the vector store.

    Indexing functionality uses a manager to keep track of which documents
    are in the vector store.

    This allows us to keep track of which documents were updated, and which
    documents were deleted, which documents should be skipped.

    For the time being, documents are indexed using their hashes, and users
     are not able to specify the uid of the document.

    Important:
       * In full mode, the loader should be returning
         the entire dataset, and not just a subset of the dataset.
         Otherwise, the auto_cleanup will remove documents that it is not
         supposed to.
       * In incremental mode, if documents associated with a particular
         source id appear across different batches, the indexing API
         will do some redundant work. This will still result in the
         correct end state of the index, but will unfortunately not be
         100% efficient. For example, if a given document is split into 15
         chunks, and we index them using a batch size of 5, we'll have 3 batches
         all with the same source id. In general, to avoid doing too much
         redundant work select as big a batch size as possible.
        * The `scoped_full` mode is suitable if determining an appropriate batch size
          is challenging or if your data loader cannot return the entire dataset at
          once. This mode keeps track of source IDs in memory, which should be fine
          for most use cases. If your dataset is large (10M+ docs), you will likely
          need to parallelize the indexing process regardless.

    Args:
        docs_source: Data loader or iterable of documents to index.
        record_manager: Timestamped set to keep track of which documents were
                         updated.
        vector_store: VectorStore or DocumentIndex to index the documents into.
        batch_size: Batch size to use when indexing. Default is 100.
        cleanup: How to handle clean up of documents. Default is None.
            - incremental: Cleans up all documents that haven't been updated AND
                           that are associated with source ids that were seen
                           during indexing.
                           Clean up is done continuously during indexing helping
                           to minimize the probability of users seeing duplicated
                           content.
            - full: Delete all documents that have not been returned by the loader
                    during this run of indexing.
                    Clean up runs after all documents have been indexed.
                    This means that users may see duplicated content during indexing.
            - scoped_full: Similar to Full, but only deletes all documents
                           that haven't been updated AND that are associated with
                           source ids that were seen during indexing.
            - None: Do not delete any documents.
        source_id_key: Optional key that helps identify the original source
            of the document. Default is None.
        cleanup_batch_size: Batch size to use when cleaning up documents.
            Default is 1_000.
        force_update: Force update documents even if they are present in the
            record manager. Useful if you are re-indexing with updated embeddings.
            Default is False.
        upsert_kwargs: Additional keyword arguments to pass to the add_documents
                       method of the VectorStore or the upsert method of the
                       DocumentIndex. For example, you can use this to
                       specify a custom vector_field:
                       upsert_kwargs={"vector_field": "embedding"}
            .. versionadded:: 0.3.10

    Returns:
        Indexing result which contains information about how many documents
        were added, updated, deleted, or skipped.

    Raises:
        ValueError: If cleanup mode is not one of 'incremental', 'full' or None
        ValueError: If cleanup mode is incremental and source_id_key is None.
        ValueError: If vectorstore does not have
            "delete" and "add_documents" required methods.
        ValueError: If source_id_key is not None, but is not a string or callable.

    .. version_modified:: 0.3.25

        * Added `scoped_full` cleanup mode.
    """
    if cleanup not in {"incremental", "full", "scoped_full", None}:
        msg = (
            f"cleanup should be one of 'incremental', 'full', 'scoped_full' or None. "
            f"Got {cleanup}."
        )
        raise ValueError(msg)

    if (cleanup == "incremental" or cleanup == "scoped_full") and source_id_key is None:
        msg = (
            "Source id key is required when cleanup mode is incremental or scoped_full."
        )
        raise ValueError(msg)

    destination = vector_store  # Renaming internally for clarity

    # If it's a vectorstore, let's check if it has the required methods.
    if isinstance(destination, VectorStore):
        # Check that the Vectorstore has required methods implemented
        methods = ["delete", "add_documents"]

        for method in methods:
            if not hasattr(destination, method):
                msg = (
                    f"Vectorstore {destination} does not have required method {method}"
                )
                raise ValueError(msg)

        if type(destination).delete == VectorStore.delete:
            # Checking if the vectorstore has overridden the default delete method
            # implementation which just raises a NotImplementedError
            msg = "Vectorstore has not implemented the delete method"
            raise ValueError(msg)
    elif isinstance(destination, DocumentIndex):
        pass
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(destination)}."
        )
        raise TypeError(msg)

    if isinstance(docs_source, BaseLoader):
        try:
            doc_iterator = docs_source.lazy_load()
        except NotImplementedError:
            doc_iterator = iter(docs_source.load())
    else:
        doc_iterator = iter(docs_source)

    source_id_assigner = _get_source_id_assigner(source_id_key)

    # Mark when the update started.
    index_start_dt = record_manager.get_time()
    num_added = 0
    num_skipped = 0
    num_updated = 0
    num_deleted = 0
    scoped_full_cleanup_source_ids: set[str] = set()

    for doc_batch in _batch(batch_size, doc_iterator):
        hashed_docs = list(
            _deduplicate_in_order(
                [_HashedDocument.from_document(doc) for doc in doc_batch]
            )
        )

        source_ids: Sequence[Optional[str]] = [
            source_id_assigner(doc) for doc in hashed_docs
        ]

        if cleanup == "incremental" or cleanup == "scoped_full":
            # source ids are required.
            for source_id, hashed_doc in zip(source_ids, hashed_docs):
                if source_id is None:
                    msg = (
                        f"Source ids are required when cleanup mode is "
                        f"incremental or scoped_full. "
                        f"Document that starts with "
                        f"content: {hashed_doc.page_content[:100]} was not assigned "
                        f"as source id."
                    )
                    raise ValueError(msg)
                if cleanup == "scoped_full":
                    scoped_full_cleanup_source_ids.add(source_id)
            # source ids cannot be None after for loop above.
            source_ids = cast("Sequence[str]", source_ids)  # type: ignore[assignment]

        exists_batch = record_manager.exists([doc.uid for doc in hashed_docs])

        # Filter out documents that already exist in the record store.
        uids = []
        docs_to_index = []
        uids_to_refresh = []
        seen_docs: set[str] = set()
        for hashed_doc, doc_exists in zip(hashed_docs, exists_batch):
            if doc_exists:
                if force_update:
                    seen_docs.add(hashed_doc.uid)
                else:
                    uids_to_refresh.append(hashed_doc.uid)
                    continue
            uids.append(hashed_doc.uid)
            docs_to_index.append(hashed_doc.to_document())

        # Update refresh timestamp
        if uids_to_refresh:
            record_manager.update(uids_to_refresh, time_at_least=index_start_dt)
            num_skipped += len(uids_to_refresh)

        # Be pessimistic and assume that all vector store write will fail.
        # First write to vector store
        if docs_to_index:
            if isinstance(destination, VectorStore):
                destination.add_documents(
                    docs_to_index,
                    ids=uids,
                    batch_size=batch_size,
                    **(upsert_kwargs or {}),
                )
            elif isinstance(destination, DocumentIndex):
                destination.upsert(
                    docs_to_index,
                    **(upsert_kwargs or {}),
                )

            num_added += len(docs_to_index) - len(seen_docs)
            num_updated += len(seen_docs)

        # And only then update the record store.
        # Update ALL records, even if they already exist since we want to refresh
        # their timestamp.
        record_manager.update(
            [doc.uid for doc in hashed_docs],
            group_ids=source_ids,
            time_at_least=index_start_dt,
        )

        # If source IDs are provided, we can do the deletion incrementally!
        if cleanup == "incremental":
            # Get the uids of the documents that were not returned by the loader.

            # mypy isn't good enough to determine that source ids cannot be None
            # here due to a check that's happening above, so we check again.
            for source_id in source_ids:
                if source_id is None:
                    msg = (
                        "source_id cannot be None at this point. "
                        "Reached unreachable code."
                    )
                    raise AssertionError(msg)

            _source_ids = cast("Sequence[str]", source_ids)

            uids_to_delete = record_manager.list_keys(
                group_ids=_source_ids, before=index_start_dt
            )
            if uids_to_delete:
                # Then delete from vector store.
                _delete(destination, uids_to_delete)
                # First delete from record store.
                record_manager.delete_keys(uids_to_delete)
                num_deleted += len(uids_to_delete)

    if cleanup == "full" or (
        cleanup == "scoped_full" and scoped_full_cleanup_source_ids
    ):
        delete_group_ids: Optional[Sequence[str]] = None
        if cleanup == "scoped_full":
            delete_group_ids = list(scoped_full_cleanup_source_ids)
        while uids_to_delete := record_manager.list_keys(
            group_ids=delete_group_ids, before=index_start_dt, limit=cleanup_batch_size
        ):
            # First delete from record store.
            _delete(destination, uids_to_delete)
            # Then delete from record manager.
            record_manager.delete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

    return {
        "num_added": num_added,
        "num_updated": num_updated,
        "num_skipped": num_skipped,
        "num_deleted": num_deleted,
    }


# Define an asynchronous generator function
async def _to_async_iterator(iterator: Iterable[T]) -> AsyncIterator[T]:
    """Convert an iterable to an async iterator."""
    for item in iterator:
        yield item


async def _adelete(
    vector_store: Union[VectorStore, DocumentIndex],
    ids: list[str],
) -> None:
    if isinstance(vector_store, VectorStore):
        delete_ok = await vector_store.adelete(ids)
        if delete_ok is not None and delete_ok is False:
            msg = "The delete operation to VectorStore failed."
            raise IndexingException(msg)
    elif isinstance(vector_store, DocumentIndex):
        delete_response = await vector_store.adelete(ids)
        if "num_failed" in delete_response and delete_response["num_failed"] > 0:
            msg = "The delete operation to DocumentIndex failed."
            raise IndexingException(msg)
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(vector_store)}."
        )
        raise TypeError(msg)


async def aindex(
    docs_source: Union[BaseLoader, Iterable[Document], AsyncIterator[Document]],
    record_manager: RecordManager,
    vector_store: Union[VectorStore, DocumentIndex],
    *,
    batch_size: int = 100,
    cleanup: Literal["incremental", "full", "scoped_full", None] = None,
    source_id_key: Union[str, Callable[[Document], str], None] = None,
    cleanup_batch_size: int = 1_000,
    force_update: bool = False,
    upsert_kwargs: Optional[dict[str, Any]] = None,
) -> IndexingResult:
    """Async index data from the loader into the vector store.

    Indexing functionality uses a manager to keep track of which documents
    are in the vector store.

    This allows us to keep track of which documents were updated, and which
    documents were deleted, which documents should be skipped.

    For the time being, documents are indexed using their hashes, and users
     are not able to specify the uid of the document.

    Important:
       * In full mode, the loader should be returning
         the entire dataset, and not just a subset of the dataset.
         Otherwise, the auto_cleanup will remove documents that it is not
         supposed to.
       * In incremental mode, if documents associated with a particular
         source id appear across different batches, the indexing API
         will do some redundant work. This will still result in the
         correct end state of the index, but will unfortunately not be
         100% efficient. For example, if a given document is split into 15
         chunks, and we index them using a batch size of 5, we'll have 3 batches
         all with the same source id. In general, to avoid doing too much
         redundant work select as big a batch size as possible.
       * The `scoped_full` mode is suitable if determining an appropriate batch size
         is challenging or if your data loader cannot return the entire dataset at
         once. This mode keeps track of source IDs in memory, which should be fine
         for most use cases. If your dataset is large (10M+ docs), you will likely
         need to parallelize the indexing process regardless.

    Args:
        docs_source: Data loader or iterable of documents to index.
        record_manager: Timestamped set to keep track of which documents were
                         updated.
        vector_store: VectorStore or DocumentIndex to index the documents into.
        batch_size: Batch size to use when indexing. Default is 100.
        cleanup: How to handle clean up of documents. Default is None.
            - incremental: Cleans up all documents that haven't been updated AND
                           that are associated with source ids that were seen
                           during indexing.
                           Clean up is done continuously during indexing helping
                           to minimize the probability of users seeing duplicated
                           content.
            - full: Delete all documents that haven to been returned by the loader.
                    Clean up runs after all documents have been indexed.
                    This means that users may see duplicated content during indexing.
            - scoped_full: Similar to Full, but only deletes all documents
                           that haven't been updated AND that are associated with
                           source ids that were seen during indexing.
            - None: Do not delete any documents.
        source_id_key: Optional key that helps identify the original source
            of the document. Default is None.
        cleanup_batch_size: Batch size to use when cleaning up documents.
            Default is 1_000.
        force_update: Force update documents even if they are present in the
            record manager. Useful if you are re-indexing with updated embeddings.
            Default is False.
        upsert_kwargs: Additional keyword arguments to pass to the aadd_documents
                       method of the VectorStore or the aupsert method of the
                       DocumentIndex. For example, you can use this to
                       specify a custom vector_field:
                       upsert_kwargs={"vector_field": "embedding"}
            .. versionadded:: 0.3.10

    Returns:
        Indexing result which contains information about how many documents
        were added, updated, deleted, or skipped.

    Raises:
        ValueError: If cleanup mode is not one of 'incremental', 'full' or None
        ValueError: If cleanup mode is incremental and source_id_key is None.
        ValueError: If vectorstore does not have
            "adelete" and "aadd_documents" required methods.
        ValueError: If source_id_key is not None, but is not a string or callable.

    .. version_modified:: 0.3.25

        * Added `scoped_full` cleanup mode.
    """
    if cleanup not in {"incremental", "full", "scoped_full", None}:
        msg = (
            f"cleanup should be one of 'incremental', 'full', 'scoped_full' or None. "
            f"Got {cleanup}."
        )
        raise ValueError(msg)

    if (cleanup == "incremental" or cleanup == "scoped_full") and source_id_key is None:
        msg = (
            "Source id key is required when cleanup mode is incremental or scoped_full."
        )
        raise ValueError(msg)

    destination = vector_store  # Renaming internally for clarity

    # If it's a vectorstore, let's check if it has the required methods.
    if isinstance(destination, VectorStore):
        # Check that the Vectorstore has required methods implemented
        # Check that the Vectorstore has required methods implemented
        methods = ["adelete", "aadd_documents"]

        for method in methods:
            if not hasattr(destination, method):
                msg = (
                    f"Vectorstore {destination} does not have required method {method}"
                )
                raise ValueError(msg)

        if type(destination).adelete == VectorStore.adelete:
            # Checking if the vectorstore has overridden the default delete method
            # implementation which just raises a NotImplementedError
            msg = "Vectorstore has not implemented the delete method"
            raise ValueError(msg)
    elif isinstance(destination, DocumentIndex):
        pass
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(destination)}."
        )
        raise TypeError(msg)
    async_doc_iterator: AsyncIterator[Document]
    if isinstance(docs_source, BaseLoader):
        try:
            async_doc_iterator = docs_source.alazy_load()
        except NotImplementedError:
            # Exception triggered when neither lazy_load nor alazy_load are implemented.
            # * The default implementation of alazy_load uses lazy_load.
            # * The default implementation of lazy_load raises NotImplementedError.
            # In such a case, we use the load method and convert it to an async
            # iterator.
            async_doc_iterator = _to_async_iterator(docs_source.load())
    else:
        if hasattr(docs_source, "__aiter__"):
            async_doc_iterator = docs_source  # type: ignore[assignment]
        else:
            async_doc_iterator = _to_async_iterator(docs_source)

    source_id_assigner = _get_source_id_assigner(source_id_key)

    # Mark when the update started.
    index_start_dt = await record_manager.aget_time()
    num_added = 0
    num_skipped = 0
    num_updated = 0
    num_deleted = 0
    scoped_full_cleanup_source_ids: set[str] = set()

    async for doc_batch in _abatch(batch_size, async_doc_iterator):
        hashed_docs = list(
            _deduplicate_in_order(
                [_HashedDocument.from_document(doc) for doc in doc_batch]
            )
        )

        source_ids: Sequence[Optional[str]] = [
            source_id_assigner(doc) for doc in hashed_docs
        ]

        if cleanup == "incremental" or cleanup == "scoped_full":
            # If the cleanup mode is incremental, source ids are required.
            for source_id, hashed_doc in zip(source_ids, hashed_docs):
                if source_id is None:
                    msg = (
                        f"Source ids are required when cleanup mode is "
                        f"incremental or scoped_full. "
                        f"Document that starts with "
                        f"content: {hashed_doc.page_content[:100]} was not assigned "
                        f"as source id."
                    )
                    raise ValueError(msg)
                if cleanup == "scoped_full":
                    scoped_full_cleanup_source_ids.add(source_id)
            # source ids cannot be None after for loop above.
            source_ids = cast("Sequence[str]", source_ids)

        exists_batch = await record_manager.aexists([doc.uid for doc in hashed_docs])

        # Filter out documents that already exist in the record store.
        uids: list[str] = []
        docs_to_index: list[Document] = []
        uids_to_refresh = []
        seen_docs: set[str] = set()
        for hashed_doc, doc_exists in zip(hashed_docs, exists_batch):
            if doc_exists:
                if force_update:
                    seen_docs.add(hashed_doc.uid)
                else:
                    uids_to_refresh.append(hashed_doc.uid)
                    continue
            uids.append(hashed_doc.uid)
            docs_to_index.append(hashed_doc.to_document())

        if uids_to_refresh:
            # Must be updated to refresh timestamp.
            await record_manager.aupdate(uids_to_refresh, time_at_least=index_start_dt)
            num_skipped += len(uids_to_refresh)

        # Be pessimistic and assume that all vector store write will fail.
        # First write to vector store
        if docs_to_index:
            if isinstance(destination, VectorStore):
                await destination.aadd_documents(
                    docs_to_index,
                    ids=uids,
                    batch_size=batch_size,
                    **(upsert_kwargs or {}),
                )
            elif isinstance(destination, DocumentIndex):
                await destination.aupsert(
                    docs_to_index,
                    **(upsert_kwargs or {}),
                )
            num_added += len(docs_to_index) - len(seen_docs)
            num_updated += len(seen_docs)

        # And only then update the record store.
        # Update ALL records, even if they already exist since we want to refresh
        # their timestamp.
        await record_manager.aupdate(
            [doc.uid for doc in hashed_docs],
            group_ids=source_ids,
            time_at_least=index_start_dt,
        )

        # If source IDs are provided, we can do the deletion incrementally!

        if cleanup == "incremental":
            # Get the uids of the documents that were not returned by the loader.

            # mypy isn't good enough to determine that source ids cannot be None
            # here due to a check that's happening above, so we check again.
            for source_id in source_ids:
                if source_id is None:
                    msg = (
                        "source_id cannot be None at this point. "
                        "Reached unreachable code."
                    )
                    raise AssertionError(msg)

            _source_ids = cast("Sequence[str]", source_ids)

            uids_to_delete = await record_manager.alist_keys(
                group_ids=_source_ids, before=index_start_dt
            )
            if uids_to_delete:
                # Then delete from vector store.
                await _adelete(destination, uids_to_delete)
                # First delete from record store.
                await record_manager.adelete_keys(uids_to_delete)
                num_deleted += len(uids_to_delete)

    if cleanup == "full" or (
        cleanup == "scoped_full" and scoped_full_cleanup_source_ids
    ):
        delete_group_ids: Optional[Sequence[str]] = None
        if cleanup == "scoped_full":
            delete_group_ids = list(scoped_full_cleanup_source_ids)
        while uids_to_delete := await record_manager.alist_keys(
            group_ids=delete_group_ids, before=index_start_dt, limit=cleanup_batch_size
        ):
            # First delete from record store.
            await _adelete(destination, uids_to_delete)
            # Then delete from record manager.
            await record_manager.adelete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

    return {
        "num_added": num_added,
        "num_updated": num_updated,
        "num_skipped": num_skipped,
        "num_deleted": num_deleted,
    }
