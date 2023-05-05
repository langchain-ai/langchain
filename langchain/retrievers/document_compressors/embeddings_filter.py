"""Document compressor that uses embeddings to drop documents unrelated to the query."""
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from pydantic import root_validator

from langchain.document_transformers import (
    _get_embeddings_from_stateful_docs,
    get_stateful_documents,
)
from langchain.embeddings.base import Embeddings
from langchain.math_utils import cosine_similarity
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.schema import Document


class EmbeddingsFilter(BaseDocumentCompressor):
    embeddings: Embeddings
    """Embeddings to use for embedding document contents and queries."""
    similarity_fn: Callable = cosine_similarity
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    k: Optional[int] = 20
    """The number of relevant documents to return. Can be set to None, in which case
    `similarity_threshold` must be specified. Defaults to 20."""
    similarity_threshold: Optional[float]
    """Threshold for determining when two documents are similar enough
    to be considered redundant. Defaults to None, must be specified if `k` is set
    to None."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_params(cls, values: Dict) -> Dict:
        """Validate similarity parameters."""
        if values["k"] is None and values["similarity_threshold"] is None:
            raise ValueError("Must specify one of `k` or `similarity_threshold`.")
        return values

    def compress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        """Filter documents based on similarity of their embeddings to the query."""
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(
            self.embeddings, stateful_documents
        )
        embedded_query = self.embeddings.embed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold
            )
            included_idxs = included_idxs[similar_enough]
        return [stateful_documents[i] for i in included_idxs]

    async def acompress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        """Filter down documents."""
        raise NotImplementedError
