"""Wrapper around DocArrayHnswSearch store."""
from __future__ import annotations

from typing import List, Optional, Type

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VST
from langchain.vectorstores.vector_store_from_doc_index import (
    VecStoreFromDocIndex,
    _check_docarray_import,
)


class DocArrayHnswSearch(VecStoreFromDocIndex):
    """Wrapper around DocArrayHnswSearch storage.

    To use it, you should have the ``docarray[hnswlib]`` package with version >=0.31.0 installed.
    You can install it with `pip install "langchain[hnswlib]"`.
    """

    def __init__(
        self,
        embedding: Embeddings,
        work_dir: str,
        n_dim: int,
        dist_metric: str = "cosine",
        max_elements: int = 1024,
        index: bool = True,
        ef_construction: int = 200,
        ef: int = 10,
        M: int = 16,
        allow_replace_deleted: bool = True,
        num_threads: int = 1,
    ) -> None:
        """Initialize DocArrayHnswSearch store.

        Args:
            embedding (Embeddings): Embedding function.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            dist_metric (str): Distance metric for DocArrayHnswSearch can be one of: "cosine",
                "ip", and "l2". Defaults to "cosine".
            max_elements (int): Maximum number of vectors that can be stored.
                Defaults to 1024.
            index (bool): Whether an index should be built for this field.
                Defaults to True.
            ef_construction (int): defines a construction time/accuracy trade-off.
                Defaults to 200.
            ef (int): parameter controlling query time/accuracy trade-off.
                Defaults to 10.
            M (int): parameter that defines the maximum number of outgoing
                connections in the graph. Defaults to 16.
            allow_replace_deleted (bool): Enables replacing of deleted elements
                with new added ones. Defaults to True.
            num_threads (int): Sets the number of cpu threads to use. Defaults to 1.
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

        doc_cls = self._get_doc_cls(
            {
                "dim": n_dim,
                "space": dist_metric,
                "max_elements": max_elements,
                "index": index,
                "ef_construction": ef_construction,
                "ef": ef,
                "M": M,
                "allow_replace_deleted": allow_replace_deleted,
                "num_threads": num_threads,
            }
        )
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
        dist_metric: str = "l2",
        max_elements: int = 1024,
        index: bool = True,
        ef_construction: int = 200,
        ef: int = 10,
        M: int = 16,
        allow_replace_deleted: bool = True,
        num_threads: int = 1,
    ) -> DocArrayHnswSearch:
        """Create an DocArrayHnswSearch store and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            dist_metric (str): Distance metric for DocArrayHnswSearch can be one of: "cosine",
                "ip", and "l2". Defaults to "l2".
            max_elements (int): Maximum number of vectors that can be stored.
                Defaults to 1024.
            index (bool): Whether an index should be built for this field.
                Defaults to True.
            ef_construction (int): defines a construction time/accuracy trade-off.
                Defaults to 200.
            ef (int): parameter controlling query time/accuracy trade-off.
                Defaults to 10.
            M (int): parameter that defines the maximum number of outgoing
                connections in the graph. Defaults to 16.
            allow_replace_deleted (bool): Enables replacing of deleted elements
                with new added ones. Defaults to True.
            num_threads (int): Sets the number of cpu threads to use. Defaults to 1.

        Returns:
            DocArrayHnswSearch Vector Store
        """
        if work_dir is None:
            raise ValueError("`work_dir` parameter has not been set.")
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
