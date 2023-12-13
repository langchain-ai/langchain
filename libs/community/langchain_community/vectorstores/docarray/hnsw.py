from __future__ import annotations

from typing import Any, List, Literal, Optional

from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.docarray.base import (
    DocArrayIndex,
    _check_docarray_import,
)


class DocArrayHnswSearch(DocArrayIndex):
    """`HnswLib` storage using `DocArray` package.

    To use it, you should have the ``docarray`` package with version >=0.32.0 installed.
    You can install it with `pip install "langchain[docarray]"`.
    """

    @classmethod
    def from_params(
        cls,
        embedding: Embeddings,
        work_dir: str,
        n_dim: int,
        dist_metric: Literal["cosine", "ip", "l2"] = "cosine",
        max_elements: int = 1024,
        index: bool = True,
        ef_construction: int = 200,
        ef: int = 10,
        M: int = 16,
        allow_replace_deleted: bool = True,
        num_threads: int = 1,
        **kwargs: Any,
    ) -> DocArrayHnswSearch:
        """Initialize DocArrayHnswSearch store.

        Args:
            embedding (Embeddings): Embedding function.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            dist_metric (str): Distance metric for DocArrayHnswSearch can be one of:
                "cosine", "ip", and "l2". Defaults to "cosine".
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
            **kwargs: Other keyword arguments to be passed to the get_doc_cls method.
        """
        _check_docarray_import()
        from docarray.index import HnswDocumentIndex

        doc_cls = cls._get_doc_cls(
            dim=n_dim,
            space=dist_metric,
            max_elements=max_elements,
            index=index,
            ef_construction=ef_construction,
            ef=ef,
            M=M,
            allow_replace_deleted=allow_replace_deleted,
            num_threads=num_threads,
            **kwargs,
        )
        doc_index = HnswDocumentIndex[doc_cls](work_dir=work_dir)  # type: ignore
        return cls(doc_index, embedding)

    @classmethod
    def from_texts(
        cls,
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

        store = cls.from_params(embedding, work_dir, n_dim, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store
