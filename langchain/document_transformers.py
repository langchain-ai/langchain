"""Transform documents"""
from typing import Any, Callable, List, Sequence

import numpy as np
from pydantic import BaseModel, Field

from langchain.embeddings.base import Embeddings
from langchain.math_utils import cosine_similarity
from langchain.schema import BaseDocumentTransformer, Document


class _DocumentWithState(Document):
    """Wrapper for a document that includes arbitrary state."""

    state: dict = Field(default_factory=dict)
    """State associated with the document."""

    def to_document(self) -> Document:
        """Convert the DocumentWithState to a Document."""
        return Document(page_content=self.page_content, metadata=self.metadata)

    @classmethod
    def from_document(cls, doc: Document) -> "_DocumentWithState":
        """Create a DocumentWithState from a Document."""
        if isinstance(doc, cls):
            return doc
        return cls(page_content=doc.page_content, metadata=doc.metadata)


def get_stateful_documents(
    documents: Sequence[Document],
) -> Sequence[_DocumentWithState]:
    return [_DocumentWithState.from_document(doc) for doc in documents]


def _filter_similar_embeddings(
    embedded_documents: List[List[float]], similarity_fn: Callable, threshold: float
) -> List[int]:
    """Filter redundant documents based on the similarity of their embeddings."""
    similarity = np.tril(similarity_fn(embedded_documents, embedded_documents), k=-1)
    redundant = np.where(similarity > threshold)
    redundant_stacked = np.column_stack(redundant)
    redundant_sorted = np.argsort(similarity[redundant])[::-1]
    included_idxs = set(range(len(embedded_documents)))
    for first_idx, second_idx in redundant_stacked[redundant_sorted]:
        if first_idx in included_idxs and second_idx in included_idxs:
            # Default to dropping the second document of any highly similar pair.
            included_idxs.remove(second_idx)
    return list(sorted(included_idxs))


def _get_embeddings_from_stateful_docs(
    embeddings: Embeddings, documents: Sequence[_DocumentWithState]
) -> List[List[float]]:
    if len(documents) and "embedded_doc" in documents[0].state:
        embedded_documents = [doc.state["embedded_doc"] for doc in documents]
    else:
        embedded_documents = embeddings.embed_documents(
            [d.page_content for d in documents]
        )
        for doc, embedding in zip(documents, embedded_documents):
            doc.state["embedded_doc"] = embedding
    return embedded_documents


class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
    """Filter that drops redundant documents by comparing their embeddings."""

    embeddings: Embeddings
    """Embeddings to use for embedding document contents."""
    similarity_fn: Callable = cosine_similarity
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    similarity_threshold: float = 0.95
    """Threshold for determining when two documents are similar enough
    to be considered redundant."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Filter down documents."""
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
