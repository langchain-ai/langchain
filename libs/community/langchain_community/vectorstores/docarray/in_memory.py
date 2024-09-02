"""Wrapper around in-memory storage."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.docarray.base import (
    DocArrayIndex,
    _check_docarray_import,
)


class DocArrayInMemorySearch(DocArrayIndex):
    """In-memory `DocArray` storage for exact search.

    To use it, you should have the ``docarray`` package with version >=0.32.0 installed.
    You can install it with `pip install docarray`.
    """

    @classmethod
    def from_params(
        cls,
        embedding: Embeddings,
        metric: Literal[
            "cosine_sim", "euclidian_dist", "sgeuclidean_dist"
        ] = "cosine_sim",
        **kwargs: Any,
    ) -> DocArrayInMemorySearch:
        """Initialize DocArrayInMemorySearch store.

        Args:
            embedding (Embeddings): Embedding function.
            metric (str): metric for exact nearest-neighbor search.
                Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".
                Defaults to "cosine_sim".
            **kwargs: Other keyword arguments to be passed to the get_doc_cls method.
        """
        _check_docarray_import()
        from docarray.index import InMemoryExactNNIndex

        doc_cls = cls._get_doc_cls(space=metric, **kwargs)
        doc_index = InMemoryExactNNIndex[doc_cls]()  # type: ignore
        return cls(doc_index, embedding)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> DocArrayInMemorySearch:
        """Create an DocArrayInMemorySearch store and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[Dict[Any, Any]]]): Metadata for each text
                if it exists. Defaults to None.
            metric (str): metric for exact nearest-neighbor search.
                Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".
                Defaults to "cosine_sim".

        Returns:
            DocArrayInMemorySearch Vector Store
        """
        store = cls.from_params(embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store
