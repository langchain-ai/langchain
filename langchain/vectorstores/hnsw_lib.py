"""Wrapper around HnswLib store."""
from __future__ import annotations

from typing import List, Optional, Type

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VST
from langchain.vectorstores.vector_store_from_doc_index import (
    VecStoreFromDocIndex,
    _check_docarray_import,
)


class HnswLib(VecStoreFromDocIndex):
    """Wrapper around HnswLib storage.

    To use it, you should have the ``docarray[hnswlib]`` package with version >=0.31.0 installed.
    You can install it with `pip install "langchain[hnswlib]"`.
    """

    def __init__(
        self,
        embedding: Embeddings,
        work_dir: str,
        n_dim: int,
        dist_metric: str = "cosine",
    ) -> None:
        """Initialize HnswLib store.

        Args:
            embedding (Embeddings): Embedding function.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            dist_metric (str): Distance metric for HnswLib can be one of: "cosine",
                "ip", and "l2". Defaults to "cosine".
        """
        _check_docarray_import()
        from docarray.index import HnswDocumentIndex

        try:
            import google.protobuf
        except ImportError:
            raise ImportError(
                "Could not import all required packages. "
                "Please install it with `pip install \"langchain[hnswlib]\"`."
            )

        doc_cls = self._get_doc_cls({"dim": n_dim, "space": dist_metric})
        doc_index = HnswDocumentIndex[doc_cls](work_dir=work_dir)
        super().__init__(doc_index, embedding)

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        work_dir: str = None,
        n_dim: int = None,
        dist_metric: str = "cosine",
    ) -> HnswLib:
        """Create an HnswLib store and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            dist_metric (str): Distance metric for HnswLib can be one of: "cosine",
                "ip", and "l2". Defaults to "cosine".

        Returns:
            HnswLib Vector Store
        """
        if work_dir is None:
            raise ValueError("`work_dir` parameter hs not been set.")
        if n_dim is None:
            raise ValueError("`n_dim` parameter has not been set.")

        store = cls(
            work_dir=work_dir,
            n_dim=n_dim,
            embedding=embedding,
            dist_metric=dist_metric,
        )
        store.add_texts(texts=texts, metadatas=metadatas)
        return store
