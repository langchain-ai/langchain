from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field, root_validator
from langchain.utils.hash import hash_nested_dict_to_uuid, hash_string_to_uuid

NAMESPACE_UUID = uuid.UUID(int=1984)


class Document(Serializable):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True


class BaseDocumentTransformer(ABC):
    """Abstract base class for document transformation systems.

    A document transformation system takes a sequence of Documents and returns a
    sequence of transformed Documents.

    Example:
        .. code-block:: python

            class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
                embeddings: Embeddings
                similarity_fn: Callable = cosine_similarity
                similarity_threshold: float = 0.95

                class Config:
                    arbitrary_types_allowed = True

                def transform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[Document]:
                    stateful_documents = get_stateful_documents(documents)
                    embedded_documents = _get_embeddings_from_stateful_docs(
                        self.embeddings, stateful_documents
                    )
                    included_idxs = _filter_similar_embeddings(
                        embedded_documents, self.similarity_fn, self.similarity_threshold
                    )
                    return [stateful_documents[i] for i in sorted(included_idxs)]

                async def atransform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[Document]:
                    raise NotImplementedError

    """  # noqa: E501

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """

    @abstractmethod
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """


class _HashedDocument(Document):
    """A hashed document with a unique ID."""

    uid: str
    hash_: str
    """The hash of the document including content and metadata."""
    content_hash: str
    """The hash of the document content."""
    metadata_hash: str
    """The hash of the document metadata."""

    @root_validator(pre=True)
    def calculate_hashes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Root validator to calculate content and metadata hash."""
        content = values.get("page_content", "")
        metadata = values.get("metadata", {})

        forbidden_keys = ("hash_", "content_hash", "metadata_hash")

        for key in forbidden_keys:
            if key in metadata:
                raise ValueError(
                    f"Metadata cannot contain key {key} as it "
                    f"is reserved for internal use."
                )

        content_hash = str(hash_string_to_uuid(content, NAMESPACE_UUID))

        try:
            metadata_hash = str(hash_nested_dict_to_uuid(metadata, NAMESPACE_UUID))
        except Exception as e:
            raise ValueError(
                f"Failed to hash metadata: {e}. "
                f"Please use a dict that can be serialized using json."
            )

        values["content_hash"] = content_hash
        values["metadata_hash"] = metadata_hash
        values["hash_"] = str(
            hash_string_to_uuid(content_hash + metadata_hash, NAMESPACE_UUID)
        )

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


def _deduplicate_in_order(
    hashed_documents: Iterable[_HashedDocument],
) -> Iterator[_HashedDocument]:
    """Deduplicate a list of hashed documents while preserving order."""
    seen = set()

    for hashed_doc in hashed_documents:
        if hashed_doc.hash_ not in seen:
            seen.add(hashed_doc.hash_)
            yield hashed_doc
