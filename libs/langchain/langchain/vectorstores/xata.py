"""Wrapper around Xata as a vector database."""

from __future__ import annotations
from itertools import repeat
from typing import (
    TYPE_CHECKING, 
    List,
    Dict,
    Any,
    Iterable,
    Optional,
    Type,
    Tuple
)

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


if TYPE_CHECKING:
    from xata.client import XataClient

class XataVectorStore(VectorStore):
    """VectorStore for a Xata database. Assumes you have a Xata database
    created with the right schema. See the guide at:
    https://integrations.langchain.com/vectorstores?integration_name=XataVectorStore

    """
    def __init__(
            self,
            client: XataClient,
            embedding: Embeddings,
            table_name: str,
    ) -> None:
        """Initialize with Xata client."""
        try:
            from xata import XataClient  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import xata python package. "
                "Please install it with `pip install xata`."
            )
        self._client = client
        self._embedding: Embeddings = embedding
        self._table_name = table_name or "vectors"

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        documents: List[Document],
        ids: List[str],
    ) -> List[str]:
        return self._add_vectors(self._client, self._table_name, vectors, documents, ids)
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        ids = ids
        docs = self._texts_to_documents(texts, metadatas)

        vectors = self._embedding.embed_documents(list(texts))
        return self.add_vectors(vectors, docs, ids)
    
    @staticmethod
    def _add_vectors(
        client: XataClient,
        table_name: str,
        vectors: List[List[float]],
        documents: List[Document],
        ids: List[str],
    ) -> List[str]:
        """Add vectors to the Xata database."""

        rows: List[Dict[str, Any]] = []
        for idx, embedding in enumerate(vectors):
                
            row = {
                    "content": documents[idx].page_content,
                    "embedding": embedding,
                }
            if ids:
                row["id"] = ids[idx]
            for key, val in documents[idx].metadata.items():
                if key not in ["id", "content", "embedding"]:
                    row[key] = val
            rows.append(row)

        # XXX: I would have liked to use the BulkProcessor here, but it
        # doesn't return the IDs, which we need here. Manual chunking it is.
        chunk_size = 1000
        id_list: List[str] = []
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i+chunk_size]

            r = client.records().bulk_insert(table_name, {"records": chunk})
            if r.status_code != 200:
                raise Exception(f"Error adding vectors to Xata: {r.status_code} {r}")
            id_list.extend(r["recordIDs"])
        return id_list
    
    @staticmethod
    def _texts_to_documents(
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[Any, Any]]] = None,
    ) -> List[Document]:
        """Return list of Documents from list of texts and metadatas."""
        if metadatas is None:
            metadatas = repeat({})

        docs = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        return docs

    @classmethod
    def from_texts(
        cls: Type["XataVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[XataClient] = None,
        table_name: Optional[str] = "vectors",
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "XataVectorStore":
        """Return VectorStore initialized from texts and embeddings."""

        if not client:
            raise ValueError("Xata client is required.")

        embeddings = embedding.embed_documents(texts)
        ids = None  # Xata will generate them for us
        docs = cls._texts_to_documents(texts, metadatas)
        cls._add_vectors(client, table_name, embeddings, docs, ids)

        return cls(
            client=client,
            embedding=embedding,
            table_name=table_name,
        )
    
    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        documents = [d[0] for d in docs_and_scores]
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[dict]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        embedding = self._embedding.embed_query(query)
        r = self._client.search_and_filter().vector_search(self._table_name, payload={
            "queryVector": embedding,
            "column": "embedding",
            "size": k,
            # "filter": filter,  # XXX: add filter support
        })
        if r.status_code != 200:
            raise Exception(f"Error running similarity search: {r.status_code} {r}")
        hits = r["records"]
        docs_and_scores = [
            (
                Document(
                    page_content=hit["content"],
                    metadata=self._extractMetadata(hit),
                ),
                hit["xata"]["score"],
            )
            for hit in hits
        ]
        return docs_and_scores
    
    def _extractMetadata(self, record: dict) -> dict:
        """Extract metadata from a record. Filters out known columns."""
        metadata = {}
        for key, val in record.items():
            if key not in ["id", "content", "embedding", "xata"]:
                metadata[key] = val
        return metadata