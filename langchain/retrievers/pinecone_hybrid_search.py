"""Taken from: https://docs.pinecone.io/docs/hybrid-search"""
import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document


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
    index: Any,
    embeddings: Embeddings,
    sparse_encoder: Any,
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[dict]] = None,
) -> None:
    """
    Create a Pinecone index from a list of contexts.
    Modifies the index argument in-place.

    Args:
        contexts: List of contexts to embed.
        index: Pinecone index to use.
        embeddings: Embeddings model to use.
        sparse_encoder: Sparse encoder to use.
        ids: List of ids to use for the documents.
        metadatas: List of metadata to use for the documents.
    """
    batch_size = 32
    _iterator = range(0, len(contexts), batch_size)
    try:
        from tqdm.auto import tqdm

        _iterator = tqdm(_iterator)
    except ImportError:
        pass

    if ids is None:
        # create unique ids using hash of the text
        ids = [hash_text(context) for context in contexts]

    for i in _iterator:
        # find end of batch
        i_end = min(i + batch_size, len(contexts))
        # extract batch
        context_batch = contexts[i:i_end]
        batch_ids = ids[i:i_end]
        metadata_batch = (
            metadatas[i:i_end] if metadatas else [{} for _ in context_batch]
        )
        # add context passages as metadata
        meta = [
            {"context": context, **metadata}
            for context, metadata in zip(context_batch, metadata_batch)
        ]

        # create dense vectors
        dense_embeds = embeddings.embed_documents(context_batch)
        # create sparse vectors
        sparse_embeds = sparse_encoder.encode_documents(context_batch)
        for s in sparse_embeds:
            s["values"] = [float(s1) for s1 in s["values"]]

        vectors = []
        # loop through the data and create dictionaries for upserts
        for doc_id, sparse, dense, metadata in zip(
            batch_ids, sparse_embeds, dense_embeds, meta
        ):
            vectors.append(
                {
                    "id": doc_id,
                    "sparse_values": sparse,
                    "values": dense,
                    "metadata": metadata,
                }
            )

        # upload the documents to the new hybrid index
        index.upsert(vectors)


class PineconeHybridSearchRetriever(BaseRetriever, BaseModel):
    embeddings: Embeddings
    sparse_encoder: Any
    index: Any
    top_k: int = 4
    alpha: float = 0.5

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        create_index(
            texts,
            self.index,
            self.embeddings,
            self.sparse_encoder,
            ids=ids,
            metadatas=metadatas,
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
            raise ValueError(
                "Could not import pinecone_text python package. "
                "Please install it with `pip install pinecone_text`."
            )
        return values

    def get_relevant_documents(self, query: str) -> List[Document]:
        from pinecone_text.hybrid import hybrid_convex_scale

        sparse_vec = self.sparse_encoder.encode_queries(query)
        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
        )
        final_result = []
        for res in result["matches"]:
            context = res["metadata"].pop("context")
            final_result.append(
                Document(page_content=context, metadata=res["metadata"])
            )
        # return search results as json
        return final_result

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
