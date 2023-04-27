"""Wrapper around in-memory DocArray store."""
from __future__ import annotations

from typing import List, Optional, Any, Tuple, Iterable, Type, Callable, Sequence, TYPE_CHECKING
from docarray.typing import NdArray

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VST
from langchain.vectorstores.vector_store_from_doc_index import VecStoreFromDocIndex, _check_docarray_import


class HnswLib(VecStoreFromDocIndex):
    """Wrapper around HnswLib storage.

    To use it, you should have the ``docarray`` package with version >=0.31.0 installed.
    """
    def __init__(
        self,
        texts: List[str],
        embedding: Embeddings,
        work_dir: str,
        n_dim: int,
        metadatas: Optional[List[dict]],
        dist_metric: str = 'cosine',
        **kwargs,
    ) -> None:
        """Initialize HnswLib store.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            work_dir (str): path to the location where all the data will be stored.
            n_dim (int): dimension of an embedding.
            dist_metric (str): Distance metric for HnswLib can be one of: 'cosine',
                'ip', and 'l2'. Defaults to 'cosine'.
        """
        _check_docarray_import()
        from docarray.index import HnswDocumentIndex

        try:
            import google.protobuf
        except ImportError:
            raise ImportError(
                "Could not import protobuf python package. "
                "Please install it with `pip install -U protobuf`."
            )

        doc_cls = self._get_doc_cls(n_dim, dist_metric)
        doc_index = HnswDocumentIndex[doc_cls](work_dir=work_dir)
        super().__init__(doc_index, texts, embedding, metadatas)

    @staticmethod
    def _get_doc_cls(n_dim: int, sim_metric: str):
        from docarray import BaseDoc
        from pydantic import Field

        class DocArrayDoc(BaseDoc):
            text: Optional[str]
            embedding: Optional[NdArray] = Field(dim=n_dim, space=sim_metric)
            metadata: Optional[dict]

        return DocArrayDoc

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        work_dir: str = None,
        n_dim: int = None,
        dist_metric: str = 'cosine',
        **kwargs: Any
    ) -> HnswLib:

        if work_dir is None:
            raise ValueError('`work_dir` parameter hs not been set.')
        if n_dim is None:
            raise ValueError('`n_dim` parameter has not been set.')

        return cls(
            work_dir=work_dir,
            n_dim=n_dim,
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            dist_metric=dist_metric,
            kwargs=kwargs,
        )
