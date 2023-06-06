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

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import (
    TypedDict,
    Sequence,
    Optional,
    Any,
    Iterator,
    Union,
    List,
    Iterable,
    Tuple,
)
from uuid import UUID

from langchain.docstore.base import ArtifactLayer, Selector
from langchain.docstore.serialization import serialize_document, deserialize_document
from langchain.document_loaders.base import BaseLoader
from langchain.output_parsers import json
from langchain.schema import Document, BaseDocumentTransformer
from langchain.text_splitter import TextSplitter

MaybeDocument = Optional[Document]

PathLike = Union[str, Path]


# TODO: MAJOR MAJOR MAJOR MAJOR
# FIX SEMANTICS WITH REGARDS TO ID, UUID. AND POTENTIALLY ARTIFACT_ID
# NEED TO REASON THROUGH USE CASES CAREFULLY TO REASON ABOUT WHATS MINIMAL SUFFICIENT


class Artifact(TypedDict):

    """A representation of an artifact."""

    custom_id: str
    """Optionally user assigned ID. If not provided, set to uuid."""
    uuid: UUID
    """A uuid represent the hash of the artifact."""
    parent_uuids: Tuple[UUID, ...]
    """A tuple of uuids representing the parent artifacts."""
    metadata: Any
    """A dictionary representing the metadata of the artifact."""


class Metadata(TypedDict):
    """Metadata format"""

    artifacts: List[Artifact]


class MetadataStore(abc.ABC):
    """Abstract metadata store."""

    def select(self, selector: Selector) -> Iterable[str]:
        """Select the artifacts matching the given selector."""
        raise NotImplementedError


class InMemoryStore(MetadataStore):
    """In-memory metadata store backed by a file.

    In its current form, this store will be really slow for large collections of files.
    """

    def __init__(self, data: Metadata) -> None:
        """Initialize the in-memory store."""
        super().__init__()
        self.data = data
        self.artifacts = data["artifacts"]
        # indexes for speed
        self.artifact_uuid = {artifact["uuid"]: artifact for artifact in self.artifacts}
        self.artifact_ids = {
            artifact["custom_id"]: artifact for artifact in self.artifacts
        }

    def exists_by_id(self, ids: Sequence[str]) -> List[bool]:
        """Order preserving check if the artifact with the given id exists."""
        return [bool(id_ in self.artifact_ids) for id_ in ids]

    def exists_by_uuid(self, uuids: Sequence[UUID]) -> List[bool]:
        """Order preserving check if the artifact with the given uuid exists."""
        return [bool(uuid_ in self.artifact_ids) for uuid_ in uuids]

    def get_by_uuids(self, uuids: Sequence[UUID]) -> List[Artifact]:
        """Return the documents with the given uuids."""
        return [self.artifact_uuid[uuid] for uuid in uuids]

    def select(self, selector: Selector) -> Iterable[UUID]:
        """Return the hashes the artifacts matching the given selector."""
        # FOR LOOP THROUGH ALL ARTIFACTS
        # Can be optimized later
        for artifact in self.data["artifacts"]:
            if selector.ids and artifact["uuid"] in selector.ids:
                yield artifact["uuid"]
                continue

            if selector.hashes and artifact["uuid"] in selector.hashes:
                yield artifact["uuid"]
                continue

            if artifact["parent_uuids"] and set(artifact["parent_uuids"]).intersection(
                selector.parent_hashes
            ):
                yield artifact["uuid"]
                continue

    def save(self, path: PathLike) -> None:
        """Save the metadata to the given path."""
        with open(path, "w") as f:
            json.dump(self.data, f)

    def add(self, artifact: Artifact) -> None:
        """Add the given artifact to the store."""
        self.data["artifacts"].append(artifact)
        # TODO(EUGENE): Handle DEFINE collision semantics
        self.artifact_uuid[artifact["uuid"]] = artifact
        self.artifact_ids[artifact["custom_id"]] = artifact

    def remove(self, selector: Selector) -> None:
        """Remove the given artifacts from the store."""
        uuids = list(self.select(selector))
        self.remove_by_uuids(uuids)

    def remove_by_uuids(self, uuids: Sequence[UUID]) -> None:
        """Remove the given artifacts from the store."""
        raise NotImplementedError()

    @classmethod
    def from_file(cls, path: PathLike) -> InMemoryStore:
        """Load store metadata from the given path."""
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
        # This is error-prone due to race conditions (if multiple
        # processes are writing), but OK for prototyping / simple use cases.
        metadata_path = root / "metadata.json"
        self.metadata_path = metadata_path
        self.metadata_store = InMemoryStore.from_file(metadata_path)

    def exists_by_uuid(self, uuids: Sequence[UUID]) -> List[bool]:
        """Check if the artifacts with the given uuid exist."""
        return self.metadata_store.exists_by_uuid(uuids)

    def exists_by_id(self, ids: Sequence[str]) -> List[bool]:
        """Check if the artifacts with the given id exist."""
        return self.metadata_store.exists_by_id(ids)

    def _get_file_path(self, uuid: UUID) -> Path:
        """Get path to file for the given uuid."""
        return self.root / f"{uuid}"

    def add(self, documents: Sequence[Document]) -> None:
        """Add the given artifacts."""
        # Write the documents to the file system
        for document in documents:
            # Use the document hash to write the contents to the file system
            file_path = self.root / f"{document.hash_}"
            with open(file_path, "w") as f:
                f.write(serialize_document(document))

            self.metadata_store.add(
                {
                    "custom_id": document.id,
                    "uuid": document.hash_,
                    "parent_uuids": document.parent_hashes,
                    "metadata": document.metadata,
                }
            )

        self.metadata_store.save(self.metadata_path)

    def get_matching_documents(self, selector: Selector) -> Iterator[Document]:
        """Can even use JQ here!"""
        uuids = self.metadata_store.select(selector)

        for uuid in uuids:
            # artifact = self.metadata_store.get_by_uuids([uuid])[0]
            path = self._get_file_path(uuid)
            with open(path, "r") as f:
                yield deserialize_document(f.read())


## VARIANT 1. And apply as a decorator
# TODO(EUGENE):
# - Figure out if it works like a decorator or a wrapper
# Likely has to work as a decorator...
# - Figure out if this inherits from BaseDocumentTransformer
# - Problem is keeping a beautiful API if we have to handle:
# -- Blob interfaces, document loading, embedding, etc.
# class CachingInterceptor:
#     """Caching interceptor for document transformers."""
#
#     def __init__(
#         self,
#         artifact_layer: ArtifactLayer,
#         underlying_transformer: BaseDocumentTransformer,
#     ) -> None:
#         """Initialize the storage interceptor."""
#         self._artifact_layer = artifact_layer
#         self._underlying_transformer = underlying_transformer
#
#     def transform(self, documents: Sequence[Document], **kwargs: Any) -> List[Document]:
#         """Transform the given documents."""
#         docs_exist = self._artifact_layer.exists_by_uuid(
#             [document.hash_ for document in documents]
#         )
#
#         # non batched variant for speed implemented
#         new_docs = []
#
#         for document, exists in zip(documents, docs_exist):
#             if not exists:
#                 transformed_docs = self._underlying_transformer.transform_documents(
#                     [document], **kwargs
#                 )
#
#                 # Make sure that lineage is included
#                 for transformed_doc in transformed_docs:
#                     if not transformed_doc.parent_hashes:
#                         transformed_doc.parent_hashes = {document.hash_}
#                 self._artifact_layer.add(transformed_docs)
#                 new_docs.extend(transformed_docs)
#             else:
#                 new_docs.extend(
#                     self._artifact_layer.get_matching_documents(
#                         Selector(parent_hashes=[document.hash_])
#                     )
#                 )
#
#         return new_docs
#


class DocumentPipeline(BaseLoader):
    def __init__(
        self,
        loader: BaseLoader,
        *,
        transformers: Optional[Sequence[BaseDocumentTransformer]] = None,
        caching_layer: Optional[ArtifactLayer] = None,
        # Would be ideal if we could implement something like this
        progress_bar: bool = False,
    ) -> None:
        """Initialize the document pipeline."""
        self.loader = loader
        self.transformers = transformers
        self.caching_layer = caching_layer

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load the documents."""
        transformations = self.transformers or []
        # Need syntax for determining whether this should be cached.
        for document in self.loader.lazy_load():
            new_documents = [document]
            for transformation in transformations:
                # Batched for now here -- logic may be a bit complex for streaming
                new_documents = list(
                    self._propagate_documents(new_documents, transformation)
                )

            yield from new_documents

    def _propagate_documents(
        self, documents: Sequence[Document], transformation: BaseDocumentTransformer
    ) -> Iterable[Document]:
        """Transform the given documents using the transformation with caching."""
        docs_exist = self.caching_layer.exists_by_uuid(
            [document.hash_ for document in documents]
        )

        # non batched variant for speed implemented
        new_docs = []

        for document, exists in zip(documents, docs_exist):
            if not exists:
                transformed_docs = transformation.transform_documents([document])

                # Make sure that lineage is included
                for transformed_doc in transformed_docs:
                    if not transformed_doc.parent_hashes:
                        transformed_doc.parent_hashes = {document.hash_}
                self.caching_layer.add(transformed_docs)
                new_docs.extend(transformed_docs)
            else:
                new_docs.extend(
                    self.caching_layer.get_matching_documents(
                        Selector(parent_hashes=[document.hash_])
                    )
                )

            yield from new_docs

    def load(self) -> List[Document]:
        """Load the documents."""
        return list(self.lazy_load())

    def execute(self) -> None:
        """Execute the pipeline."""
        for _ in self.lazy_load():
            pass

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        raise NotImplementedError("This method will never be implemented.")
