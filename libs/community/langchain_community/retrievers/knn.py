"""KNN Retriever.
Largely based on
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Iterable, List, Optional

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


def create_index(contexts: List[str], embeddings: Embeddings) -> np.ndarray:
    """
    Create an index of embeddings for a list of contexts.

    Args:
        contexts: List of contexts to embed.
        embeddings: Embeddings model to use.

    Returns:
        Index of embeddings.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return np.array(list(executor.map(embeddings.embed_query, contexts)))


class KNNRetriever(BaseRetriever):
    """`KNN` retriever."""

    embeddings: Embeddings
    """Embeddings model to use."""
    index: Any
    """Index of embeddings."""
    texts: List[str]
    """List of texts to index."""
    metadatas: Optional[List[dict]] = None
    """List of metadatas corresponding with each text."""
    k: int = 4
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None
    """Threshold for relevancy."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embeddings: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> KNNRetriever:
        index = create_index(texts, embeddings)
        return cls(
            embeddings=embeddings,
            index=index,
            texts=texts,
            metadatas=metadatas,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> KNNRetriever:
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts, embeddings=embeddings, metadatas=metadatas, **kwargs
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = [
            Document(
                page_content=self.texts[row],
                metadata=self.metadatas[row] if self.metadatas else {},
            )
            for row in sorted_ix[0 : self.k]
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            )
        ]
        return top_k_results
