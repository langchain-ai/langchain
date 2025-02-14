from typing import Callable, Dict, Optional, Sequence

import numpy as np
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import ConfigDict, Field

from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)


def _get_similarity_function() -> Callable:
    try:
        from langchain_community.utils.math import cosine_similarity
    except ImportError:
        raise ImportError(
            "To use please install langchain-community "
            "with `pip install langchain-community`."
        )
    return cosine_similarity


class EmbeddingsFilter(BaseDocumentCompressor):
    """Document compressor that uses embeddings to drop documents
    unrelated to the query."""

    embeddings: Embeddings
    """Embeddings to use for embedding document contents and queries."""
    similarity_fn: Callable = Field(default_factory=_get_similarity_function)
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    k: Optional[int] = 20
    """The number of relevant documents to return. Can be set to None, in which case
    `similarity_threshold` must be specified. Defaults to 20."""
    similarity_threshold: Optional[float] = None
    """Threshold for determining when two documents are similar enough
    to be considered redundant. Defaults to None, must be specified if `k` is set
    to None."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @pre_init
    def validate_params(cls, values: Dict) -> Dict:
        """Validate similarity parameters."""
        if values["k"] is None and values["similarity_threshold"] is None:
            raise ValueError("Must specify one of `k` or `similarity_threshold`.")
        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter documents based on similarity of their embeddings to the query."""
        try:
            from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E501
                _get_embeddings_from_stateful_docs,
                get_stateful_documents,
            )
        except ImportError:
            raise ImportError(
                "To use please install langchain-community "
                "with `pip install langchain-community`."
            )
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(
            self.embeddings, stateful_documents
        )
        embedded_query = self.embeddings.embed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs: np.ndarray = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold
            )
            included_idxs = included_idxs[similar_enough]
        for i in included_idxs:
            stateful_documents[i].state["query_similarity_score"] = similarity[i]
        return [stateful_documents[i] for i in included_idxs]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter documents based on similarity of their embeddings to the query."""
        try:
            from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E501
                _aget_embeddings_from_stateful_docs,
                get_stateful_documents,
            )
        except ImportError:
            raise ImportError(
                "To use please install langchain-community "
                "with `pip install langchain-community`."
            )
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = await _aget_embeddings_from_stateful_docs(
            self.embeddings, stateful_documents
        )
        embedded_query = await self.embeddings.aembed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs: np.ndarray = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold
            )
            included_idxs = included_idxs[similar_enough]
        for i in included_idxs:
            stateful_documents[i].state["query_similarity_score"] = similarity[i]
        return [stateful_documents[i] for i in included_idxs]
