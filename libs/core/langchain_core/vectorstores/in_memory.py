from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumpd, load
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import _cosine_similarity as cosine_similarity
from langchain_core.vectorstores.utils import (
    _maximal_marginal_relevance as maximal_marginal_relevance,
)

if TYPE_CHECKING:
    from langchain_core.indexing import UpsertResponse


class InMemoryVectorStore(VectorStore):
    """In-memory implementation of VectorStore using a dictionary.

    Uses numpy to compute cosine similarity for search.
    """

    def __init__(self, embedding: Embeddings) -> None:
        """Initialize with the given embedding function.

        Args:
            embedding: embedding function to use.
        """
        # TODO: would be nice to change to
        # Dict[str, Document] at some point (will be a breaking change)
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

    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        vectors = self.embedding.embed_documents([item.page_content for item in items])
        ids = []
        for item, vector in zip(items, vectors):
            doc_id = item.id if item.id else str(uuid.uuid4())
            ids.append(doc_id)
            self.store[doc_id] = {
                "id": doc_id,
                "vector": vector,
                "text": item.page_content,
                "metadata": item.metadata,
            }
        return {
            "succeeded": ids,
            "failed": [],
        }

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        documents = []

        for doc_id in ids:
            doc = self.store.get(doc_id)
            if doc:
                documents.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=doc["metadata"],
                    )
                )
        return documents

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Async get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        return self.get_by_ids(ids)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self.add_texts(texts, metadatas, **kwargs)

    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, List[float]]]:
        result = []
        for doc in self.store.values():
            vector = doc["vector"]
            similarity = float(cosine_similarity([embedding], [vector]).item(0))
            result.append(
                (
                    Document(
                        id=doc["id"], page_content=doc["text"], metadata=doc["metadata"]
                    ),
                    similarity,
                    vector,
                )
            )
        result.sort(key=lambda x: x[1], reverse=True)
        if filter is not None:
            result = [r for r in result if filter(r[0])]
        return result[:k]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, filter=filter, **kwargs
            )
        ]

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
            **kwargs,
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
            **kwargs,
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
        prefetch_hits = self._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            **kwargs,
        )

        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "numpy must be installed to use max_marginal_relevance_search "
                "pip install numpy"
            )

        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [vector for _, _, vector in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        return [prefetch_hits[idx][0] for idx in mmr_chosen_indices]

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
            **kwargs,
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

    @classmethod
    def load(
        cls, path: str, embedding: Embeddings, **kwargs: Any
    ) -> "InMemoryVectorStore":
        """Load a vector store from a file.

        Args:
            path: The path to load the vector store from.
            embedding: The embedding to use.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            A VectorStore object.
        """
        _path: Path = Path(path)
        with _path.open("r") as f:
            store = load(json.load(f))
        vectorstore = cls(embedding=embedding, **kwargs)
        vectorstore.store = store
        return vectorstore

    def dump(self, path: str) -> None:
        """Dump the vector store to a file.

        Args:
            path: The path to dump the vector store to.
        """
        _path: Path = Path(path)
        _path.parent.mkdir(exist_ok=True, parents=True)
        with _path.open("w") as f:
            json.dump(dumpd(self.store), f, indent=2)
