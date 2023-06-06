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

from langchain.docstore.base import ArtifactStore, Selector
from langchain.docstore.serialization import serialize_document, deserialize_document
from langchain.document_loaders.base import BaseLoader
from langchain.output_parsers import json
from langchain.schema import Document, BaseDocumentTransformer
from langchain.text_splitter import TextSplitter

MaybeDocument = Optional[Document]

PathLike = Union[str, Path]


# KNOWN WAYS THIS CAN FAIL:
# 1) If the process crashes while text splitting, creating only some of the artifacts
#    ... new pipeline will not re-create the missing artifacts! (at least for now)
#    it will use the ones that exist and assume that all of them have been created

# TODO: MAJOR MAJOR MAJOR MAJOR
# FIX SEMANTICS WITH REGARDS TO ID, UUID. AND POTENTIALLY ARTIFACT_ID
# NEED TO REASON THROUGH USE CASES CAREFULLY TO REASON ABOUT WHATS MINIMAL SUFFICIENT


class Artifact(TypedDict):

    """A representation of an artifact."""

    uid: str
    """A unique identifier for the artifact."""
    parent_uids: Tuple[str, ...]
    """A tuple of uuids representing the parent artifacts."""
    metadata: Any
    """A dictionary representing the metadata of the artifact."""
    tags: Tuple[str, ...]
    """A tuple of tags associated with the artifact.
    
    Can use tags to add information about the transformation that was applied
    to the given artifact. There's probably a better representation.
    """


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
        self.artifact_uids = {artifact["uid"]: artifact for artifact in self.artifacts}

    def exists_by_uids(self, uids: Sequence[str]) -> List[bool]:
        """Order preserving check if the artifact with the given id exists."""
        return [bool(uid in self.artifact_uids) for uid in uids]

    def get_by_uids(self, uids: Sequence[str]) -> List[Artifact]:
        """Return the documents with the given uuids."""
        return [self.artifact_uids[uid] for uid in uids]

    def select(self, selector: Selector) -> Iterable[str]:
        """Return the hashes the artifacts matching the given selector."""
        # FOR LOOP THROUGH ALL ARTIFACTS
        # Can be optimized later
        for artifact in self.data["artifacts"]:
            uid = artifact["uid"]
            # Implement conjunctive normal form
            if selector.uids and artifact["uid"] in selector.uids:
                yield uid
                continue

            if artifact["parent_uids"] and set(artifact["parent_uids"]).intersection(
                selector.parent_uids
            ):
                yield uid
                continue

    def save(self, path: PathLike) -> None:
        """Save the metadata to the given path."""
        with open(path, "w") as f:
            json.dump(self.data, f)

    def add(self, artifact: Artifact) -> None:
        """Add the given artifact to the store."""
        self.data["artifacts"].append(artifact)
        # TODO(EUGENE): Handle DEFINE collision semantics
        self.artifact_uids[artifact["uid"]] = artifact

    def remove(self, selector: Selector) -> None:
        """Remove the given artifacts from the store."""
        uids = list(self.select(selector))
        self.remove_by_uuids(uids)

    def remove_by_uuids(self, uids: Sequence[str]) -> None:
        """Remove the given artifacts from the store."""
        for uid in uids:
            del self.artifact_uids[uid]

    @classmethod
    def from_file(cls, path: PathLike) -> InMemoryStore:
        """Load store metadata from the given path."""
        with open(path, "r") as f:
            content = json.load(f)
        return cls(content)


class FileSystemArtifactLayer(ArtifactStore):
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

    def exists_by_uid(self, uuids: Sequence[str]) -> List[bool]:
        """Check if the artifacts with the given uuid exist."""
        return self.metadata_store.exists_by_uids(uuids)

    def _get_file_path(self, uuid: UUID) -> Path:
        """Get path to file for the given uuid."""
        return self.root / f"{uuid}"

    def add(self, documents: Sequence[Document], tags: Sequence[str]) -> None:
        """Add the given artifacts."""
        # Write the documents to the file system
        for document in documents:
            # Use the document hash to write the contents to the file system
            file_path = self.root / f"{document.hash_}"
            with open(file_path, "w") as f:
                f.write(serialize_document(document))

            self.metadata_store.add(
                {
                    "uid": document.uid,
                    "parent_uids": document.parent_uids,
                    "metadata": document.metadata,
                    "tags": tags,
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
        artifact_store: Optional[ArtifactStore] = None,
        # Would be ideal if we could implement something like this
        progress_bar: bool = False,
    ) -> None:
        """Initialize the document pipeline."""
        self.loader = loader
        self.transformers = transformers
        self.artifact_store = artifact_store

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
        docs_exist = self.artifact_store.exists_by_uid(
            [document.uid for document in documents]
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

                # TODO(EUGENE): Extract transformation information here
                # to add to metadata store
                transformation_name = transformation.__class__.__name__
                self.artifact_store.add(transformed_docs, tags=[transformation_name])
                new_docs.extend(transformed_docs)
            else:
                new_docs.extend(
                    # TODO(Eugene): In this form, this will fail in cases
                    # where the original transformation failed half way throough
                    # and only created some of the children.
                    # MAJOR
                    self.artifact_store.get_matching_documents(
                        Selector(parent_uids=[document.uid])
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
