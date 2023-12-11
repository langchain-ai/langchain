from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import DistanceStrategy


class SemaDB(VectorStore):
    """`SemaDB` vector store.

    This vector store is a wrapper around the SemaDB database.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import SemaDB

            db = SemaDB('mycollection', 768, embeddings, DistanceStrategy.COSINE)

    """

    HOST = "semadb.p.rapidapi.com"
    BASE_URL = "https://" + HOST

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        api_key: str = "",
    ):
        """Initialise the SemaDB vector store."""
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.api_key = api_key or get_from_env("api_key", "SEMADB_API_KEY")
        self._embedding = embedding
        self.distance_strategy = distance_strategy

    @property
    def headers(self) -> dict:
        """Return the common headers."""
        return {
            "content-type": "application/json",
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": SemaDB.HOST,
        }

    def _get_internal_distance_strategy(self) -> str:
        """Return the internal distance strategy."""
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return "euclidean"
        elif self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            raise ValueError("Max inner product is not supported by SemaDB")
        elif self.distance_strategy == DistanceStrategy.DOT_PRODUCT:
            return "dot"
        elif self.distance_strategy == DistanceStrategy.JACCARD:
            raise ValueError("Max inner product is not supported by SemaDB")
        elif self.distance_strategy == DistanceStrategy.COSINE:
            return "cosine"
        else:
            raise ValueError(f"Unknown distance strategy {self.distance_strategy}")

    def create_collection(self) -> bool:
        """Creates the corresponding collection in SemaDB."""
        payload = {
            "id": self.collection_name,
            "vectorSize": self.vector_size,
            "distanceMetric": self._get_internal_distance_strategy(),
        }
        response = requests.post(
            SemaDB.BASE_URL + "/collections",
            json=payload,
            headers=self.headers,
        )
        return response.status_code == 200

    def delete_collection(self) -> bool:
        """Deletes the corresponding collection in SemaDB."""
        response = requests.delete(
            SemaDB.BASE_URL + f"/collections/{self.collection_name}",
            headers=self.headers,
        )
        return response.status_code == 200

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        if not isinstance(texts, list):
            texts = list(texts)
        embeddings = self._embedding.embed_documents(texts)
        # Check dimensions
        if len(embeddings[0]) != self.vector_size:
            raise ValueError(
                f"Embedding size mismatch {len(embeddings[0])} != {self.vector_size}"
            )
        # Normalise if needed
        if self.distance_strategy == DistanceStrategy.COSINE:
            embed_matrix = np.array(embeddings)
            embed_matrix = embed_matrix / np.linalg.norm(
                embed_matrix, axis=1, keepdims=True
            )
            embeddings = embed_matrix.tolist()
        # Create points
        ids: List[str] = []
        points = []
        if metadatas is not None:
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                new_id = str(uuid4())
                ids.append(new_id)
                points.append(
                    {
                        "id": new_id,
                        "vector": embedding,
                        "metadata": {**metadata, **{"text": text}},
                    }
                )
        else:
            for text, embedding in zip(texts, embeddings):
                new_id = str(uuid4())
                ids.append(new_id)
                points.append(
                    {
                        "id": new_id,
                        "vector": embedding,
                        "metadata": {"text": text},
                    }
                )
        # Insert points in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            response = requests.post(
                SemaDB.BASE_URL + f"/collections/{self.collection_name}/points",
                json={"points": batch},
                headers=self.headers,
            )
            if response.status_code != 200:
                print("HERE--", batch)
                raise ValueError(f"Error adding points: {response.text}")
            failed_ranges = response.json()["failedRanges"]
            if len(failed_ranges) > 0:
                raise ValueError(f"Error adding points: {failed_ranges}")
        # Return ids
        return ids

    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings."""
        return self._embedding

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        payload = {
            "ids": ids,
        }
        response = requests.delete(
            SemaDB.BASE_URL + f"/collections/{self.collection_name}/points",
            json=payload,
            headers=self.headers,
        )
        return response.status_code == 200 and len(response.json()["failedPoints"]) == 0

    def _search_points(self, embedding: List[float], k: int = 4) -> List[dict]:
        """Search points."""
        # Normalise if needed
        if self.distance_strategy == DistanceStrategy.COSINE:
            vec = np.array(embedding)
            vec = vec / np.linalg.norm(vec)
            embedding = vec.tolist()
        # Perform search request
        payload = {
            "vector": embedding,
            "limit": k,
        }
        response = requests.post(
            SemaDB.BASE_URL + f"/collections/{self.collection_name}/points/search",
            json=payload,
            headers=self.headers,
        )
        if response.status_code != 200:
            raise ValueError(f"Error searching: {response.text}")
        return response.json()["points"]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        query_embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        query_embedding = self._embedding.embed_query(query)
        points = self._search_points(query_embedding, k=k)
        return [
            (
                Document(page_content=p["metadata"]["text"], metadata=p["metadata"]),
                p["distance"],
            )
            for p in points
        ]

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        points = self._search_points(embedding, k=k)
        return [
            Document(page_content=p["metadata"]["text"], metadata=p["metadata"])
            for p in points
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "",
        vector_size: int = 0,
        api_key: str = "",
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        **kwargs: Any,
    ) -> "SemaDB":
        """Return VectorStore initialized from texts and embeddings."""
        if not collection_name:
            raise ValueError("Collection name must be provided")
        if not vector_size:
            raise ValueError("Vector size must be provided")
        if not api_key:
            raise ValueError("API key must be provided")
        semadb = cls(
            collection_name,
            vector_size,
            embedding,
            distance_strategy=distance_strategy,
            api_key=api_key,
        )
        if not semadb.create_collection():
            raise ValueError("Error creating collection")
        semadb.add_texts(texts, metadatas=metadatas)
        return semadb
