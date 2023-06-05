"""Base persistence layer for artifacts.

This code makes a few assumptions:

1) Vector stores can accept a STRING user provided ID for a document and store the document.
2) We can fit all the document IDs into memory
3) Existing transformers operate on [doc] -> [doc] and would need to be updated to keep track of history  (parent_doc_hashes)
4) Changing the transformer interface to operate on doc -> doc or doc -> [doc], will allow the an interceptor to update the history by itself.


Here are some possible APIs for this (we would want to converge to the simplest correct version)

Usage:

    ... code-block:: python

    file_system_store = FileSystemArtifactLayer( # <-- All artifacts will be stored here
        parent_dir=Path("data/artifacts"),
    )
    
    pipeline = sequential(
        [MimeParser(), TextSplitter()], interceptor=CachingDocumentTransformer(file_system_store)
    )
    
    doc_iterable = FileSystemLoader.from("data/my_videos", pipeline)
    vector_store = VectorStore.from(doc_iterable)
    
    
## Or some variations
    
    pipeline = compose_transformation(
        [MimeParser(), TextSplitter(), VectorStore.from], interceptor=CachingDocumentTransformer(file_system_store)
    )
    
    
## or
    
    ... code-block:: python

    file_system_store = FileSystemArtifactLayer( # <-- All artifacts will be stored here
        parent_dir=Path("data/artifacts"),
    )
    
    pipeline = sequential(
        [MimeParser(), TextSplitter()], interceptor=CachingDocumentTransformer(file_system_store)
    )
    
    
    _ = pipeline.process(docs) # <-- This will store the docs in the file system store
    
    sync(
        file_system_store, vector_store, selector={
            "provenance": startswith("https://wikipedia"), # All content from wikipedia
            "parent_transformer": "TextSplitter", # After content was text splitted
            "updated_after": today().offset(hours=-5) # updated in the last 5 hours
        }
    ) # <-- This will sync the file system store with the vector store
"""
import dataclasses
from typing import TypedDict, Sequence, Optional, Any, Iterator
from uuid import UUID
from pathlib import Path

from langchain.docstore.base import ArtifactLayer
from langchain.output_parsers import json
from langchain.schema import Document, BaseDocumentTransformer
from langchain.vectorstores.base import VectorStore

MaybeDocument = Optional[Document]


@dataclasses.dataclass(frozen=True)
class BaseSelector:
    pass


@dataclasses.dataclass(frozen=True)
class Selector(BaseSelector):
    """Selector for documents."""

    parent: Optional[UUID] = None
    provenance: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class FreeFormSelector(BaseSelector):
    # Some selector that matches syntax of underlying vector store
    query: str
    kwargs: Any


def serialize_document(document: Document) -> str:
    """Serialize the given document to a string."""
    raise NotImplementedError()


def deserialize_document(serialized_document: str) -> Document:
    """Deserialize the given document from a string."""
    raise NotImplementedError()


class FileSystemArtifactLayer(ArtifactLayer):
    def __init__(self, parent_dir: Path) -> None:
        self.parent_dir = parent_dir
        # Bad, but keep JSON file memory for now (race conditions/locks etc)
        metadata_path = parent_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata_json = json.load(f)
        self.metadata_json = metadata_json

    def exists(self, ids: Sequence[str]) -> Sequence[bool]:
        """Check if the artifacts with the given id exist."""
        # Use the metadata file to check if the file exists

    def add(self, documents: Sequence[Document]):
        """Add the given artifacts."""
        # Write the documents to the file system
        for document in documents:
            # Use the document hash to write the contents to the file system
            with open(self.parent_dir / f"{document.hash_}", "w") as f:
                f.write(serialize_document(document))

    def get_matching_documents(self, selector: Selector) -> Iterator[Document]:
        """Can even use JQ here!"""
        # Use the metadata file to get the matching documents with the selector
        matching_document_uuids = []

        for uuid in matching_document_uuids:
            with open(self.parent_dir / f"{uuid}", "r") as f:
                yield deserialize_document(f.read())


class CachingDocumentTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        artifact_layer: ArtifactLayer,
        # This wraps a particular transformer
        # Once hashes are added to the transformation logic itself
        # We can skip the usage of the transformer completely if transformation
        # and content hashes match
        document_transformer: BaseDocumentTransformer,
    ) -> None:
        """Initialize the storage interceptor."""
        self._artifact_layer = artifact_layer
        self._document_transformer = document_transformer

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform the given documents."""
        existence = self._artifact_layer.exists([document.id for document in documents])

        # non batched variant for speed implemented
        new_docs = []

        for document, exists in zip(documents, existence):
            if not exists:
                transformed_docs = self._document_transformer.transform_documents(
                    [document], **kwargs
                )
                self._artifact_layer.add(transformed_docs)
                new_docs.extend(transformed_docs)
            else:
                new_docs.extend(
                    self._artifact_layer.get_child_documents(document.hash_)
                )

        return new_docs


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
