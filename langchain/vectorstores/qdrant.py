"""Wrapper around Qdrant vector database."""
import uuid
from operator import itemgetter
from typing import Any, Callable, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


class Qdrant(VectorStore):
    """Wrapper around Qdrant vector database.

    To use you should have the ``qdrant-client`` package installed.

    Example:
        .. code-block:: python

            from langchain import Qdrant

            client = QdrantClient()
            collection_name = "MyCollection"
            qdrant = Qdrant(client, collection_name, embedding_function)
    """

    def __init__(self, client: Any, collection_name: str, embedding_function: Callable):
        """Initialize with necessary components."""
        try:
            import qdrant_client
        except ImportError:
            raise ValueError(
                "Could not import qdrant-client python package. "
                "Please it install it with `pip install qdrant-client`."
            )

        if not isinstance(client, qdrant_client.QdrantClient):
            raise ValueError(
                f"client should be an instance of qdrant_client.QdrantClient, "
                f"got {type(client)}"
            )

        self.client: qdrant_client.QdrantClient = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        from qdrant_client.http import models as rest

        ids = [uuid.uuid4().hex for _ in texts]
        self.client.upsert(
            collection_name=self.collection_name,
            points=rest.Batch(
                ids=ids,
                vectors=[self.embedding_function(text) for text in texts],
                payloads=self._build_payloads(texts, metadatas),
            ),
        )

        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        results = self.similarity_search_with_score(query, k)
        return list(map(itemgetter(0), results))

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            with_payload=True,
            limit=k,
        )
        return [
            (
                self._document_from_scored_point(result),
                result.score,
            )
            for result in results
        ]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            with_payload=True,
            with_vectors=True,
            limit=k,
        )
        embeddings = [result.vector for result in results]
        mmr_selected = maximal_marginal_relevance(embedding, embeddings, k=k)
        return [self._document_from_scored_point(results[i]) for i in mmr_selected]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "Qdrant":
        """Construct Qdrant wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the Qdrant database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Qdrant
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                qdrant = Qdrant.from_texts(texts, embeddings)
        """
        try:
            import qdrant_client
        except ImportError:
            raise ValueError(
                "Could not import qdrant-client python package. "
                "Please it install it with `pip install qdrant-client`."
            )

        from qdrant_client.http import models as rest

        # Just do a single quick embedding to get vector size
        embeddings = embedding.embed_documents(texts[:1])
        vector_size = len(embeddings[0])

        qdrant_host = get_from_dict_or_env(kwargs, "host", "QDRANT_HOST")
        kwargs.pop("host")
        client = qdrant_client.QdrantClient(host=qdrant_host, **kwargs)

        collection_name = kwargs.get("collection_name", uuid.uuid4().hex)
        distance_func = kwargs.pop("distance_func", "Cosine").upper()
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance[distance_func],
            ),
        )

        client.upsert(
            collection_name=collection_name,
            points=rest.Batch(
                ids=[uuid.uuid4().hex for _ in texts],
                vectors=embeddings,
                payloads=cls._build_payloads(texts, metadatas),
            ),
        )

        return cls(client, collection_name, embedding.embed_query)

    @classmethod
    def _build_payloads(
        cls, texts: Iterable[str], metadatas: Optional[List[dict]]
    ) -> List[dict]:
        return [
            {
                "page_content": text,
                "metadata": metadatas[i] if metadatas is not None else None,
            }
            for i, text in enumerate(texts)
        ]

    @classmethod
    def _document_from_scored_point(cls, scored_point: Any) -> Document:
        return Document(
            page_content=scored_point.payload.get("page_content"),
            metadata=scored_point.payload.get("metadata") or {},
        )
