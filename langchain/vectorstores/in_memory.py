"""Wrapper around in-memory storage."""
from __future__ import annotations

from typing import List, Optional, Type

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VST
from langchain.vectorstores.vector_store_from_doc_index import (
    VecStoreFromDocIndex,
    _check_docarray_import,
)


class InMemory(VecStoreFromDocIndex):
    """Wrapper around in-memory storage.

    To use it, you should have the ``docarray`` package with version >=0.31.0 installed.
    You can install it with `pip install "langchain[in_memory_store]"`.
    """

    def __init__(
        self,
        embedding: Embeddings,
        metric: str = "cosine_sim",
    ) -> None:
        """Initialize in-memory store.

        Args:
            embedding (Embeddings): Embedding function.
            metric (str): metric for exact nearest-neighbor search.
                Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".
                Defaults to "cosine_sim".
        """
        _check_docarray_import()
        from docarray.index import InMemoryExactNNIndex

        doc_cls = self._get_doc_cls({"space": metric})
        doc_index = InMemoryExactNNIndex[doc_cls]()
        super().__init__(doc_index, embedding)

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = "cosine_sim",
    ) -> InMemory:
        """Create an in-memory store and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            metric (str): metric for exact nearest-neighbor search.
                Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".
                Defaults to "cosine_sim".

        Returns:
            InMemory Vector Store
        """
        store = cls(
            embedding=embedding,
            metric=metric,
        )
        store.add_texts(texts=texts, metadatas=metadatas)
        return store
