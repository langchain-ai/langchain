from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from zep_cloud import CreateDocumentRequest, DocumentCollectionResponse, SearchType

logger = logging.getLogger()


class ZepCloudVectorStore(VectorStore):
    """`Zep` vector store.

    It provides methods for adding texts or documents to the store,
    searching for similar documents, and deleting documents.

    Search scores are calculated using cosine similarity normalized to [0, 1].

    Args:
        collection_name (str): The name of the collection in the Zep store.
        api_key (str): The API key for the Zep API.
    """

    def __init__(
        self,
        collection_name: str,
        api_key: str,
    ) -> None:
        super().__init__()
        if not collection_name:
            raise ValueError(
                "collection_name must be specified when using ZepVectorStore."
            )
        try:
            from zep_cloud.client import AsyncZep, Zep
        except ImportError:
            raise ImportError(
                "Could not import zep-python python package. "
                "Please install it with `pip install zep-python`."
            )
        self._client = Zep(api_key=api_key)
        self._client_async = AsyncZep(api_key=api_key)

        self.collection_name = collection_name

        self._load_collection()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Unavailable for ZepCloud"""
        return None

    def _load_collection(self) -> DocumentCollectionResponse:
        """
        Load the collection from the Zep backend.
        """
        from zep_cloud import NotFoundError

        try:
            collection = self._client.document.get_collection(self.collection_name)
        except NotFoundError:
            logger.info(
                f"Collection {self.collection_name} not found. Creating new collection."
            )
            collection = self._create_collection()

        return collection

    def _create_collection(self) -> DocumentCollectionResponse:
        """
        Create a new collection in the Zep backend.
        """
        self._client.document.add_collection(self.collection_name)
        collection = self._client.document.get_collection(self.collection_name)
        return collection

    def _generate_documents_to_add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[CreateDocumentRequest]:
        from zep_cloud import CreateDocumentRequest as ZepDocument

        documents: List[ZepDocument] = []
        for i, d in enumerate(texts):
            documents.append(
                ZepDocument(
                    content=d,
                    metadata=metadatas[i] if metadatas else None,
                    document_id=document_ids[i] if document_ids else None,
                )
            )
        return documents

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            document_ids: Optional list of document ids associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        documents = self._generate_documents_to_add(texts, metadatas, document_ids)
        uuids = self._client.document.add_documents(
            self.collection_name, request=documents
        )

        return uuids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        documents = self._generate_documents_to_add(texts, metadatas, document_ids)
        uuids = await self._client_async.document.add_documents(
            self.collection_name, request=documents
        )

        return uuids

    def search(
        self,
        query: str,
        search_type: SearchType,
        metadata: Optional[Dict[str, Any]] = None,
        k: int = 3,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query using specified search type."""
        if search_type == "similarity":
            return self.similarity_search(query, k=k, metadata=metadata, **kwargs)
        elif search_type == "mmr":
            return self.max_marginal_relevance_search(
                query, k=k, metadata=metadata, **kwargs
            )
        else:
            raise ValueError(
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity' or 'mmr'."
            )

    async def asearch(
        self,
        query: str,
        search_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        k: int = 3,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query using specified search type."""
        if search_type == "similarity":
            return await self.asimilarity_search(
                query, k=k, metadata=metadata, **kwargs
            )
        elif search_type == "mmr":
            return await self.amax_marginal_relevance_search(
                query, k=k, metadata=metadata, **kwargs
            )
        else:
            raise ValueError(
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity' or 'mmr'."
            )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""

        results = self._similarity_search_with_relevance_scores(
            query, k=k, metadata=metadata, **kwargs
        )
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""

        return self._similarity_search_with_relevance_scores(
            query, k=k, metadata=metadata, **kwargs
        )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            metadata: Optional, metadata filter
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 and
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """

        results = self._client.document.search(
            collection_name=self.collection_name,
            text=query,
            limit=k,
            metadata=metadata,
            **kwargs,
        )

        return [
            (
                Document(
                    page_content=str(doc.content),
                    metadata=doc.metadata,
                ),
                doc.score or 0.0,
            )
            for doc in results.results or []
        ]

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""

        results = await self._client_async.document.search(
            collection_name=self.collection_name,
            text=query,
            limit=k,
            metadata=metadata,
            **kwargs,
        )

        return [
            (
                Document(
                    page_content=str(doc.content),
                    metadata=doc.metadata,
                ),
                doc.score or 0.0,
            )
            for doc in results.results or []
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""

        results = await self.asimilarity_search_with_relevance_scores(
            query, k, metadata=metadata, **kwargs
        )

        return [doc for doc, _ in results]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Unsupported in Zep Cloud"""
        warnings.warn("similarity_search_by_vector is not supported in Zep Cloud")
        return []

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Unsupported in Zep Cloud"""
        warnings.warn("asimilarity_search_by_vector is not supported in Zep Cloud")
        return []

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Zep determines this automatically and this parameter is
                        ignored.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            metadata: Optional, metadata to filter the resulting set of retrieved docs
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        results = self._client.document.search(
            collection_name=self.collection_name,
            text=query,
            limit=k,
            metadata=metadata,
            search_type="mmr",
            mmr_lambda=lambda_mult,
            **kwargs,
        )

        return [
            Document(page_content=str(d.content), metadata=d.metadata)
            for d in results.results or []
        ]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""

        results = await self._client_async.document.search(
            collection_name=self.collection_name,
            text=query,
            limit=k,
            metadata=metadata,
            search_type="mmr",
            mmr_lambda=lambda_mult,
            **kwargs,
        )

        return [
            Document(page_content=str(d.content), metadata=d.metadata)
            for d in results.results or []
        ]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Unsupported in Zep Cloud"""
        warnings.warn(
            "max_marginal_relevance_search_by_vector is not supported in Zep Cloud"
        )
        return []

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Unsupported in Zep Cloud"""
        warnings.warn(
            "amax_marginal_relevance_search_by_vector is not supported in Zep Cloud"
        )
        return []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ZepCloudVectorStore:
        """
        Class method that returns a ZepVectorStore instance initialized from texts.

        If the collection does not exist, it will be created.

        Args:
            texts (List[str]): The list of texts to add to the vectorstore.
            metadatas (Optional[List[Dict[str, Any]]]): Optional list of metadata
               associated with the texts.
            collection_name (str): The name of the collection in the Zep store.
            api_key (str): The API key for the Zep API.
            **kwargs: Additional parameters specific to the vectorstore.

        Returns:
            ZepVectorStore: An instance of ZepVectorStore.
        """
        if not api_key:
            raise ValueError("api_key must be specified when using ZepVectorStore.")
        vecstore = cls(
            collection_name=collection_name,
            api_key=api_key,
        )
        vecstore.add_texts(texts, metadatas)
        return vecstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by Zep vector UUIDs.

        Parameters
        ----------
        ids : Optional[List[str]]
            The UUIDs of the vectors to delete.

        Raises
        ------
        ValueError
            If no UUIDs are provided.
        """

        if ids is None or len(ids) == 0:
            raise ValueError("No uuids provided to delete.")

        for u in ids:
            self._client.document.delete_document(self.collection_name, u)
