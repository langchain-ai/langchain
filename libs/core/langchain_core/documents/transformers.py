"""Document transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document


class BaseDocumentTransformer(ABC):
    """Abstract base class for document transformation.

    A document transformation takes a sequence of Documents and returns a
    sequence of transformed Documents.

    Example:
        ```python
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
                    embedded_documents,
                    self.similarity_fn,
                    self.similarity_threshold,
                )
                return [stateful_documents[i] for i in sorted(included_idxs)]

            async def atransform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                raise NotImplementedError
        ```
    """

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of `Document` objects to be transformed.

        Returns:
            A sequence of transformed `Document` objects.
        """

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of `Document` objects to be transformed.

        Returns:
            A sequence of transformed `Document` objects.
        """
        return await run_in_executor(
            None, self.transform_documents, documents, **kwargs
        )
