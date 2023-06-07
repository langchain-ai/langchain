"""Implement artifact storage using the file system.

This is a simple implementation that stores artifacts in a directory and
metadata in a JSON file. It's used for prototyping.

Metadata should move into a SQLLite.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import (
    TypedDict,
    Sequence,
    Optional,
    Iterator,
    Union,
    List,
    Iterable,
)

from langchain.docstore.base import ArtifactStore, Selector, Artifact, ArtifactWithData
from langchain.docstore.serialization import serialize_document, deserialize_document
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

MaybeDocument = Optional[Document]

PathLike = Union[str, Path]


class Metadata(TypedDict):
    """Metadata format"""

    artifacts: List[Artifact]


class MetadataStore(abc.ABC):
    """Abstract metadata store.

    Need to populate with all required methods.
    """

    @abc.abstractmethod
    def upsert(self, artifact: Artifact):
        """Add the given artifact to the store."""

    @abc.abstractmethod
    def select(self, selector: Selector) -> Iterable[str]:
        """Select the artifacts matching the given selector."""
        raise NotImplementedError


class CacheBackedEmbedder:
    """Interface for embedding models."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        underlying_embedder: Embeddings,
    ) -> None:
        """Initialize the embedder."""
        self.artifact_store = artifact_store
        self.underlying_embedder = underlying_embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        raise NotImplementedError()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        raise NotImplementedError()


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
        # Inefficient implementation that loops through all artifacts.
        # Optimize later.
        for artifact in self.data["artifacts"]:
            uid = artifact["uid"]
            # Implement conjunctive normal form
            if selector.uids and artifact["uid"] in selector.uids:
                yield uid
                continue

            if selector.parent_uids and set(artifact["parent_uids"]).intersection(
                selector.parent_uids
            ):
                yield uid
                continue

    def save(self, path: PathLike) -> None:
        """Save the metadata to the given path."""
        with open(path, "w") as f:
            json.dump(self.data, f)

    def upsert(self, artifact: Artifact) -> None:
        """Add the given artifact to the store."""
        uid = artifact["uid"]
        if uid not in self.artifact_uids:
            self.data["artifacts"].append(artifact)
            self.artifact_uids[artifact["uid"]] = artifact

    def remove(self, selector: Selector) -> None:
        """Remove the given artifacts from the store."""
        uids = list(self.select(selector))
        self.remove_by_uuids(uids)

    def remove_by_uuids(self, uids: Sequence[str]) -> None:
        """Remove the given artifacts from the store."""
        for uid in uids:
            del self.artifact_uids[uid]
        raise NotImplementedError(f"Need to delete artifacts as well")

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
        _root = root if isinstance(root, Path) else Path(root)
        self.root = _root
        # Metadata file will be kept in memory for now and updated with
        # each call.
        # This is error-prone due to race conditions (if multiple
        # processes are writing), but OK for prototyping / simple use cases.
        metadata_path = _root / "metadata.json"
        self.metadata_path = metadata_path

        if metadata_path.exists():
            self.metadata_store = InMemoryStore.from_file(self.metadata_path)
        else:
            self.metadata_store = InMemoryStore({"artifacts": []})

    def exists_by_uid(self, uuids: Sequence[str]) -> List[bool]:
        """Check if the artifacts with the given uuid exist."""
        return self.metadata_store.exists_by_uids(uuids)

    def _get_file_path(self, uid: str) -> Path:
        """Get path to file for the given uuid."""
        return self.root / f"{uid}"

    def upsert(
        self,
        artifacts_with_data: Sequence[ArtifactWithData],
    ) -> None:
        """Add the given artifacts."""
        # Write the documents to the file system
        for artifact_with_data in artifacts_with_data:
            # Use the document hash to write the contents to the file system
            document = artifact_with_data["document"]
            file_path = self.root / f"{document.hash_}"
            with open(file_path, "w") as f:
                f.write(serialize_document(document))

            artifact = artifact_with_data["artifact"].copy()
            # Storing at a file -- can clean up the artifact with data request
            # later
            artifact["location"] = str(file_path)
            self.metadata_store.upsert(artifact)

        self.metadata_store.save(self.metadata_path)

    def list_document_ids(self, selector: Selector) -> Iterator[str]:
        """List the document ids matching the given selector."""
        yield from self.metadata_store.select(selector)

    def list_documents(self, selector: Selector) -> Iterator[Document]:
        """Can even use JQ here!"""
        uuids = self.metadata_store.select(selector)

        for uuid in uuids:
            artifact = self.metadata_store.get_by_uids([uuid])[0]
            path = artifact["location"]
            with open(path, "r") as f:
                page_content = deserialize_document(f.read()).page_content
                yield Document(
                    uid=artifact["uid"],
                    parent_uids=artifact["parent_uids"],
                    metadata=artifact["metadata"],
                    tags=artifact["tags"],
                    page_content=page_content,
                )
