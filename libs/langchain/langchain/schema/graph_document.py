from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field
from langchain.schema import Document


class Node(Serializable):
    """Class for storing a piece of text and associated metadata."""

    id: Union[str, int]
    type: str = "Node"
    properties: dict = Field(default_factory=dict)


class Relationship(Serializable):
    """Class for storing a piece of text and associated metadata."""

    source: Node
    target: Node
    type: str
    properties: dict = Field(default_factory=dict)


class GraphDocument(Serializable):
    """Class for storing a piece of text and associated metadata."""

    nodes: List[Node]
    """String text."""
    relationships: List[Relationship]
    source: Document
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """


class BaseGraphDocumentTransformer(ABC):
    """Abstract base class for graph document transformation systems.

    A document transformation system takes a sequence of Documents and returns a
    sequence of graph Documents.

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
    ) -> Sequence[GraphDocument]:
        """Transform a list of documents to graph documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """

    @abstractmethod
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[GraphDocument]:
        """Asynchronously transform a list of documents to graph documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
