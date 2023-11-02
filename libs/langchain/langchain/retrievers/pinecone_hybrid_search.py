"""Taken from: https://docs.pinecone.io/docs/hybrid-search"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BaseRetriever, Document
from langchain.schema.embeddings import Embeddings
from langchain.utils.iter import batch_iterate

if TYPE_CHECKING:
    from pinecone import Index
from abc import ABCMeta, abstractmethod


class PineconeIndexUpsert(metaclass=ABCMeta):
    """
    An interface for upserting vectors into a Pinecone index.
    """

    def __init__(self, index: Index):
        self.index = index

    @abstractmethod
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 64,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Upsert vectors into a Pinecone index.

        Args:
            vectors: List of vectors to upsert.
            namespace: Namespace of the index.
            batch_size: Batch size for upserting.

        Returns:
            None
        """
        ...

    def __call__(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 64,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return self.upsert(vectors=vectors, namespace=namespace, batch_size=batch_size)

    @classmethod
    def get_index_upsert(
        cls, index_name: str, pool_threads: int = 1
    ) -> PineconeIndexUpsert:
        """Get an instance of PineconeIndexUpsert.

        It is a wrapper around pinecone.Index that provides an
        interface for upserting vectors either synchronously or using threads.
        """
        import pinecone

        ret: Optional[PineconeIndexUpsert] = None
        if pool_threads > 1:
            index = pinecone.Index(index_name, pool_threads=pool_threads)
            ret = ThreadedIndexUpsert(index)
        else:
            index = pinecone.Index(index_name)
            ret = SyncIndexUpsert(index)
        return ret


class ThreadedIndexUpsert(PineconeIndexUpsert):
    """Upsert vectors into a Pinecone index using threads."""

    def __init__(self, index: Index) -> None:
        super().__init__(index)

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 64,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # A threaded parallel implementation of upserting vectors into a Pinecone index.
        # It works only for REST API, not gRPC.
        async_res = [
            self.index.upsert(
                batch, namespace=namespace, async_req=True, *args, **kwargs
            )
            for batch in batch_iterate(batch_size, vectors)
        ]
        [res.get() for res in async_res]


class SyncIndexUpsert(PineconeIndexUpsert):
    """Upsert vectors into a Pinecone index synchronously."""

    def __init__(self, index: Index) -> None:
        super().__init__(index)

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 64,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.index.upsert(
            vectors, namespace=namespace, batch_size=batch_size, *args, **kwargs
        )


def hash_text(text: str) -> str:
    """Hash a text using SHA256.

    Args:
        text: Text to hash.

    Returns:
        Hashed text.
    """
    return str(hashlib.sha256(text.encode("utf-8")).hexdigest())


def create_index(
    contexts: List[str],
    index_upsert: PineconeIndexUpsert,
    embeddings: Embeddings,
    sparse_encoder: Any,
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[dict]] = None,
    namespace: Optional[str] = None,
    batch_size: int = 32,
    chunk_size: int = 1000,
) -> None:
    """Create an index from a list of contexts.

    It modifies the index argument in-place!

    Args:
        contexts: List of contexts to embed.
        index_upsert: PineconeIndexUpsert instance to use for upserting.
        embeddings: Embeddings model to use.
        sparse_encoder: Sparse encoder to use.
        ids: List of ids to use for the documents.
        metadatas: List of metadata to use for the documents.
        namespace: Namespace to use for the documents.
        batch_size: Batch size to use for the index upsert.
        chunk_size: Chunk size to use for the calculating embeddings.
    """
    # get index upsert threaded or not
    _iterator = range(0, len(contexts), chunk_size)
    try:
        from tqdm.auto import tqdm

        _iterator = tqdm(_iterator)
    except ImportError:
        pass

    if ids is None:
        # create unique ids using hash of the text
        ids = [hash_text(context) for context in contexts]

    metadatas = metadatas or [{} for _ in contexts]
    for metadata, context in zip(metadatas, contexts):
        metadata["context"] = context

    for i in _iterator:
        # extract batch
        chunk_batch = contexts[i : i + chunk_size]
        chunk_ids = ids[i : i + chunk_size]
        chunk_metadata = metadatas[i : i + chunk_size]

        # create dense vectors
        dense_embeds = embeddings.embed_documents(chunk_batch)
        # create sparse vectors
        sparse_embeds = sparse_encoder.encode_documents(chunk_batch)
        for s in sparse_embeds:
            s["values"] = [float(s1) for s1 in s["values"]]

        vectors = [
            {
                "id": doc_id,
                "sparse_values": sparse,
                "values": dense,
                "metadata": metadata,
            }
            for doc_id, sparse, dense, metadata in zip(
                chunk_ids, sparse_embeds, dense_embeds, chunk_metadata
            )
        ]
        index_upsert.upsert(vectors, namespace=namespace, batch_size=batch_size)


class PineconeHybridSearchRetriever(BaseRetriever):
    """`Pinecone Hybrid Search` retriever."""

    embeddings: Embeddings
    """Embeddings model to use."""
    """description"""
    sparse_encoder: Any
    """Sparse encoder to use."""
    index_upsert: PineconeIndexUpsert
    """Pinecone index to use."""
    top_k: int = 4
    """Number of documents to return."""
    alpha: float = 0.5
    """Alpha value for hybrid search."""
    namespace: Optional[str] = None
    """Namespace value for index partition."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        namespace: Optional[str] = None,
        batch_size: int = 32,
        chunk_size: int = 1000,
    ) -> None:
        create_index(
            texts,
            self.index_upsert,
            self.embeddings,
            self.sparse_encoder,
            ids=ids,
            metadatas=metadatas,
            namespace=namespace,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from pinecone_text.hybrid import hybrid_convex_scale  # noqa:F401
            from pinecone_text.sparse.base_sparse_encoder import (
                BaseSparseEncoder,  # noqa:F401
            )
        except ImportError:
            raise ImportError(
                "Could not import pinecone_text python package. "
                "Please install it with `pip install pinecone_text`."
            )
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        from pinecone_text.hybrid import hybrid_convex_scale

        sparse_vec = self.sparse_encoder.encode_queries(query)
        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        result = self.index_upsert.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        final_result = []
        for res in result["matches"]:
            context = res["metadata"].pop("context")
            final_result.append(
                Document(page_content=context, metadata=res["metadata"])
            )
        # return search results as json
        return final_result
