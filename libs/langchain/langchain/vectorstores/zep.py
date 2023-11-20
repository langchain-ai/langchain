from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

if TYPE_CHECKING:
    from zep_python.document import Document as ZepDocument
    from zep_python.document import DocumentCollection


logger = logging.getLogger()


@dataclass
class CollectionConfig:
    """Configuration for a `Zep Collection`.

    If the collection does not exist, it will be created.

    Attributes:
        name (str): The name of the collection.
        description (Optional[str]): An optional description of the collection.
        metadata (Optional[Dict[str, Any]]): Optional metadata for the collection.
        embedding_dimensions (int): The number of dimensions for the embeddings in
            the collection. This should match the Zep server configuration
            if auto-embed is true.
        is_auto_embedded (bool): A flag indicating whether the collection is
            automatically embedded by Zep.
    """

    name: str
    description: Optional[str]
    metadata: Optional[Dict[str, Any]]
    embedding_dimensions: int
    is_auto_embedded: bool


class ZepVectorStore(VectorStore):
    """`Zep` vector store.

    It provides methods for adding texts or documents to the store,
    searching for similar documents, and deleting documents.

    Search scores are calculated using cosine similarity normalized to [0, 1].

    Args:
        api_url (str): The URL of the Zep API.
        collection_name (str): The name of the collection in the Zep store.
        api_key (Optional[str]): The API key for the Zep API.
        config (Optional[CollectionConfig]): The configuration for the collection.
            Required if the collection does not already exist.
        embedding (Optional[Embeddings]): Optional embedding function to use to
            embed the texts. Required if the collection is not auto-embedded.
    """

    def __init__(
        self,
        collection_name: str,
        api_url: str,
        *,
        api_key: Optional[str] = None,
        config: Optional[CollectionConfig] = None,
        embedding: Optional[Embeddings] = None,
    ) -> None:
        super().__init__()
        if not collection_name:
            raise ValueError(
                "collection_name must be specified when using ZepVectorStore."
            )
        try:
            from zep_python import ZepClient
        except ImportError:
            raise ImportError(
                "Could not import zep-python python package. "
                "Please install it with `pip install zep-python`."
            )
        self._client = ZepClient(api_url, api_key=api_key)

        self.collection_name = collection_name
        # If for some reason the collection name is not the same as the one in the
        # config, update it.
        if config and config.name != self.collection_name:
            config.name = self.collection_name

        self._collection_config = config
        self._collection = self._load_collection()
        self._embedding = embedding

        # self.add_texts(texts, metadatas=metadatas, **kwargs)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embedding

    def _load_collection(self) -> DocumentCollection:
        """
        Load the collection from the Zep backend.
        """
        from zep_python import NotFoundError

        try:
            collection = self._client.document.get_collection(self.collection_name)
        except NotFoundError:
            logger.info(
                f"Collection {self.collection_name} not found. Creating new collection."
            )
            collection = self._create_collection()

        return collection

    def _create_collection(self) -> DocumentCollection:
        """
        Create a new collection in the Zep backend.
        """
        if not self._collection_config:
            raise ValueError(
                "Collection config must be specified when creating a new collection."
            )
        collection = self._client.document.add_collection(
            **asdict(self._collection_config)
        )
        return collection

    def _generate_documents_to_add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[ZepDocument]:
        from zep_python.document import Document as ZepDocument

        embeddings = None
        if self._collection and self._collection.is_auto_embedded:
            if self._embedding is not None:
                warnings.warn(
                    """The collection is set to auto-embed and an embedding 
                function is present. Ignoring the embedding function.""",
                    stacklevel=2,
                )
        elif self._embedding is not None:
            embeddings = self._embedding.embed_documents(list(texts))
            if self._collection and self._collection.embedding_dimensions != len(
                embeddings[0]
            ):
                raise ValueError(
                    "The embedding dimensions of the collection and the embedding"
                    " function do not match. Collection dimensions:"
                    f" {self._collection.embedding_dimensions}, Embedding dimensions:"
                    f" {len(embeddings[0])}"
                )
        else:
            pass

        documents: List[ZepDocument] = []
        for i, d in enumerate(texts):
            documents.append(
                ZepDocument(
                    content=d,
                    metadata=metadatas[i] if metadatas else None,
                    document_id=document_ids[i] if document_ids else None,
                    embedding=embeddings[i] if embeddings else None,
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
        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        documents = self._generate_documents_to_add(texts, metadatas, document_ids)
        uuids = self._collection.add_documents(documents)

        return uuids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        documents = self._generate_documents_to_add(texts, metadatas, document_ids)
        uuids = await self._collection.aadd_documents(documents)

        return uuids

    def search(
        self,
        query: str,
        search_type: str,
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

        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if not self._collection.is_auto_embedded and self._embedding:
            query_vector = self._embedding.embed_query(query)
            results = self._collection.search(
                embedding=query_vector, limit=k, metadata=metadata, **kwargs
            )
        else:
            results = self._collection.search(
                query, limit=k, metadata=metadata, **kwargs
            )

        return [
            (
                Document(
                    page_content=doc.content,
                    metadata=doc.metadata,
                ),
                doc.score or 0.0,
            )
            for doc in results
        ]

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""

        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if not self._collection.is_auto_embedded and self._embedding:
            query_vector = self._embedding.embed_query(query)
            results = await self._collection.asearch(
                embedding=query_vector, limit=k, metadata=metadata, **kwargs
            )
        else:
            results = await self._collection.asearch(
                query, limit=k, metadata=metadata, **kwargs
            )

        return [
            (
                Document(
                    page_content=doc.content,
                    metadata=doc.metadata,
                ),
                doc.score or 0.0,
            )
            for doc in results
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
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            metadata: Optional, metadata filter

        Returns:
            List of Documents most similar to the query vector.
        """
        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        results = self._collection.search(
            embedding=embedding, limit=k, metadata=metadata, **kwargs
        )

        return [
            Document(
                page_content=doc.content,
                metadata=doc.metadata,
            )
            for doc in results
        ]

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector."""
        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        results = self._collection.search(
            embedding=embedding, limit=k, metadata=metadata, **kwargs
        )

        return [
            Document(
                page_content=doc.content,
                metadata=doc.metadata,
            )
            for doc in results
        ]

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

        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if not self._collection.is_auto_embedded and self._embedding:
            query_vector = self._embedding.embed_query(query)
            results = self._collection.search(
                embedding=query_vector,
                limit=k,
                metadata=metadata,
                search_type="mmr",
                mmr_lambda=lambda_mult,
                **kwargs,
            )
        else:
            results, query_vector = self._collection.search_return_query_vector(
                query,
                limit=k,
                metadata=metadata,
                search_type="mmr",
                mmr_lambda=lambda_mult,
                **kwargs,
            )

        return [Document(page_content=d.content, metadata=d.metadata) for d in results]

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

        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if not self._collection.is_auto_embedded and self._embedding:
            query_vector = self._embedding.embed_query(query)
            results = await self._collection.asearch(
                embedding=query_vector,
                limit=k,
                metadata=metadata,
                search_type="mmr",
                mmr_lambda=lambda_mult,
                **kwargs,
            )
        else:
            results, query_vector = await self._collection.asearch_return_query_vector(
                query,
                limit=k,
                metadata=metadata,
                search_type="mmr",
                mmr_lambda=lambda_mult,
                **kwargs,
            )

        return [Document(page_content=d.content, metadata=d.metadata) for d in results]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
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
            embedding: Embedding to look up documents similar to.
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
        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        results = self._collection.search(
            embedding=embedding,
            limit=k,
            metadata=metadata,
            search_type="mmr",
            mmr_lambda=lambda_mult,
            **kwargs,
        )

        return [Document(page_content=d.content, metadata=d.metadata) for d in results]

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        if not self._collection:
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        results = await self._collection.asearch(
            embedding=embedding,
            limit=k,
            metadata=metadata,
            search_type="mmr",
            mmr_lambda=lambda_mult,
            **kwargs,
        )

        return [Document(page_content=d.content, metadata=d.metadata) for d in results]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "",
        api_url: str = "",
        api_key: Optional[str] = None,
        config: Optional[CollectionConfig] = None,
        **kwargs: Any,
    ) -> ZepVectorStore:
        """
        Class method that returns a ZepVectorStore instance initialized from texts.

        If the collection does not exist, it will be created.

        Args:
            texts (List[str]): The list of texts to add to the vectorstore.
            embedding (Optional[Embeddings]): Optional embedding function to use to
               embed the texts.
            metadatas (Optional[List[Dict[str, Any]]]): Optional list of metadata
               associated with the texts.
            collection_name (str): The name of the collection in the Zep store.
            api_url (str): The URL of the Zep API.
            api_key (Optional[str]): The API key for the Zep API.
            config (Optional[CollectionConfig]): The configuration for the collection.
            **kwargs: Additional parameters specific to the vectorstore.

        Returns:
            ZepVectorStore: An instance of ZepVectorStore.
        """
        vecstore = cls(
            collection_name,
            api_url,
            api_key=api_key,
            config=config,
            embedding=embedding,
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

        if self._collection is None:
            raise ValueError("No collection name provided.")

        for u in ids:
            self._collection.delete_document(u)
