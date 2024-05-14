import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.utils.math import cosine_similarity
from langchain_community.vectorstores.utils import maximal_marginal_relevance


class InMemoryVectorStore(VectorStore):
    """In-memory implementation of VectorStore using a dictionary.
    Uses numpy to compute cosine similarity for search.

    Args:
        embedding:  embedding function to use.
    """

    def __init__(self, embedding: Embeddings) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self.embedding = embedding

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def delete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    async def adelete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        self.delete(ids)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the store."""
        vectors = self.embedding.embed_documents(list(texts))
        ids_ = []

        for i, text in enumerate(texts):
            doc_id = ids[i] if ids else str(uuid.uuid4())
            ids_.append(doc_id)
            self.store[doc_id] = {
                "id": doc_id,
                "vector": vectors[i],
                "text": text,
                "metadata": metadatas[i] if metadatas else {},
            }
        return ids_

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self.add_texts(texts, metadatas, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        docs_with_similarity = []
        for doc in self.store.values():
            similarity = float(cosine_similarity([embedding], [doc["vector"]]).item(0))
            docs_with_similarity.append(
                (
                    Document(page_content=doc["text"], metadata=doc["metadata"]),
                    similarity,
                )
            )
        docs_with_similarity.sort(key=lambda x: x[1], reverse=True)
        return docs_with_similarity[:k]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
        )
        return docs

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score(query, k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.similarity_search(query, k, **kwargs)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        docs_with_similarity = []
        for doc in self.store.values():
            similarity = float(cosine_similarity([embedding], [doc["vector"]]).item(0))
            docs_with_similarity.append(
                (
                    doc,
                    similarity,
                )
            )
        docs_with_similarity.sort(key=lambda x: x[1], reverse=True)
        prefetch_hits = docs_with_similarity[:fetch_k]

        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [doc["vector"] for doc, _ in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        return [
            Document(
                page_content=prefetch_hits[idx][0]["text"],
                metadata=prefetch_hits[idx][0]["metadata"],
            )
            for idx in mmr_chosen_indices
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "InMemoryVectorStore":
        store = cls(
            embedding=embedding,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "InMemoryVectorStore":
        return cls.from_texts(texts, embedding, metadatas, **kwargs)
