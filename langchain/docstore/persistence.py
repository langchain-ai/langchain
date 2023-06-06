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
from json import JSONDecodeError

import abc
import json
from pathlib import Path
from typing import TypedDict, Sequence, Optional, Any, Iterator, Union, List, Iterable
from uuid import UUID

from langchain.docstore.base import ArtifactLayer, Selector
from langchain.output_parsers import json
from langchain.schema import Document, BaseDocumentTransformer
from langchain.vectorstores.base import VectorStore

MaybeDocument = Optional[Document]


class UUIDEncoder(json.JSONEncoder):
    """TODO detemine if there's a better solution."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)  # Convert UUID to string
        return super().default(obj)


def serialize_document(document: Document) -> str:
    """Serialize the given document to a string."""
    try:
        return json.dumps(document.dict(), cls=UUIDEncoder)
    except JSONDecodeError:
        raise ValueError(f"Could not serialize document with ID: {document.id}")


def deserialize_document(serialized_document: str) -> Document:
    """Deserialize the given document from a string."""
    return Document.parse_obj(json.loads(serialized_document))


PathLike = Union[str, Path]


class Artifact(TypedDict):
    id: str
    hash_: str
    parent_hashes: List[str]
    metadata: Any


class MetadataFormat(TypedDict):
    artifacts: List[Artifact]


class MetadataStore(abc.ABC):
    """Abstract metadata store."""

    def select(self, selector: Selector) -> Sequence[str]:
        """Select the artifacts matching the given selector."""
        raise NotImplementedError


class InMemoryStore(MetadataStore):
    def __init__(self, data: MetadataFormat) -> None:
        """Initialize the in-memory store."""
        self.data = data

    def select(self, selector: Selector) -> Iterable[str]:
        """Return an iterable of the matching artifacts."""
        # FOR LOOP THROUGH ALL ARTIFACTS. THIS IS FINE
        # It's done to avoid keeping 

        for artifact in self.data:


        already_seen = set()
        if selector.parent_hashes:
            # Non efficient
            for artifact in self.data["artifacts"]:
                if set(artifact["parent_hashes"]).intersection(selector.parent_hashes):
                    yield artifact["id"]
                    already_seen.add(artifact["id"])

        if selector.ids:
            for artifact in self.data["artifacts"]:
                if (
                    artifact["id"] in selector.ids
                    and artifact["id"] not in already_seen
                ):
                    yield artifact["id"]
                    already_seen.add(artifact["id"])

        if selector.hashes:
            for artifact in self.data["artifacts"]:
                if (
                    artifact["hash_"] in selector.hashes
                    and artifact["id"] not in already_seen
                ):
                    yield artifact["id"]
                    already_seen.add(artifact["id"])

    def save(self, path: PathLike) -> None:
        """Save the metadata to the given path."""
        with open(path, "w") as f:
            json.dump(self.data, f)

    def add(self, artifact: Artifact) -> None:
        """Add the given artifact to the store."""
        self.data["artifacts"].append(artifact)

    def remove(self, selector: Selector) -> None:
        """Remove the given artifacts from the store."""
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: PathLike) -> "InMemoryStore":
        with open(path, "r") as f:
            content = json.load(f)

        return cls(content)


class FileSystemArtifactLayer(ArtifactLayer):
    """An artifact layer for storing artifacts on the file system."""

    def __init__(self, root: PathLike) -> None:
        """Initialize the file system artifact layer."""
        self.root = root if isinstance(root, Path) else Path(root)
        # Metadata file will be kept in memory for now and updated with
        # each call.
        # This is hacky and error prone due to race conditions (if multiple
        # processes are writing), but OK for prototyping.
        metadata_path = root / "metadata.json"
        self.metadata_store = InMemoryStore.from_file(metadata_path)

    def exists(self, ids: Sequence[str]) -> Sequence[bool]:
        """Check if the artifacts with the given id exist."""
        # Use the metadata file to check if the file exists

    def add(self, documents: Sequence[Document]):
        """Add the given artifacts."""
        # Write the documents to the file system
        for document in documents:
            # Use the document hash to write the contents to the file system
            file_path = self.root / f"{document.hash_}"
            with open(file_path, "w") as f:
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
