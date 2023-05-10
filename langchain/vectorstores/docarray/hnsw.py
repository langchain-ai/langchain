"""Wrapper around Hnswlib store."""
from __future__ import annotations

from typing import Any, List, Optional, Type

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.docarray.base import DocArrayIndex, _check_docarray_import


class DocArrayHnswSearch(DocArrayIndex):
    """Wrapper around HnswLib storage.

    To use it, you should have the ``docarray[hnswlib]`` package with version >=0.31.0
    installed. You can install it with `pip install "langchain[hnswlib]"`.
    """

    def __init__(
        self, embedding: Embeddings, work_dir: str, n_dim: int, **kwargs: Any
    ) -> None:
        """Initialize DocArrayHnswSearch store.

        Args:
            embedding (Embeddings): Embedding function.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            **kwargs: Other keyword arguments to be passed to the _get_doc_cls method.
        """
        _check_docarray_import()
        from docarray.index import HnswDocumentIndex

        kwargs.setdefault("dist_metric", "cosine")
        kwargs.setdefault("max_elements", 1024)
        kwargs.setdefault("index", True)
        kwargs.setdefault("ef_construction", 200)
        kwargs.setdefault("ef", 10)
        kwargs.setdefault("M", 16)
        kwargs.setdefault("allow_replace_deleted", True)
        kwargs.setdefault("num_threads", 1)

        doc_cls = self._get_doc_cls(
            {
                "dim": n_dim,
                "space": kwargs["dist_metric"],
                **{k: v for k, v in kwargs.items() if k != "dist_metric"},
            }
        )
        doc_index = HnswDocumentIndex[doc_cls](work_dir=work_dir)  # type: ignore
        super().__init__(doc_index, embedding)

    @classmethod
    def from_texts(
        cls: Type[DocArrayHnswSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        work_dir: Optional[str] = None,
        n_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> DocArrayHnswSearch:
        """Create an DocArrayHnswSearch store and insert data.


        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            **kwargs: Other keyword arguments to be passed to the __init__ method.

        Returns:
            DocArrayHnswSearch Vector Store
        """
        if work_dir is None:
            raise ValueError("`work_dir` parameter has not been set.")
        if n_dim is None:
            raise ValueError("`n_dim` parameter has not been set.")

        store = cls(work_dir=work_dir, n_dim=n_dim, embedding=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store
