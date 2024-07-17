from __future__ import annotations

import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

if TYPE_CHECKING:
    import bagel
    import bagel.config
    from bagel.api.types import ID, OneOrMany, Where, WhereDocument

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore

DEFAULT_K = 5


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class Bagel(VectorStore):
    """``Bagel.net`` Inference platform.

    To use, you should have the ``bagelML`` python package installed.

    Example:
        .. code-block:: python

                from langchain_community.vectorstores import Bagel
                vectorstore = Bagel(cluster_name="langchain_store")
    """

    _LANGCHAIN_DEFAULT_CLUSTER_NAME = "langchain"

    def __init__(
        self,
        cluster_name: str = _LANGCHAIN_DEFAULT_CLUSTER_NAME,
        client_settings: Optional[bagel.config.Settings] = None,
        embedding_function: Optional[Embeddings] = None,
        cluster_metadata: Optional[Dict] = None,
        client: Optional[bagel.Client] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Initialize with bagel client"""
        try:
            import bagel
            import bagel.config
        except ImportError:
            raise ImportError("Please install bagel `pip install bagelML`.")
        if client is not None:
            self._client_settings = client_settings
            self._client = client
        else:
            if client_settings:
                _client_settings = client_settings
            else:
                _client_settings = bagel.config.Settings(
                    bagel_api_impl="rest",
                    bagel_server_host="api.bageldb.ai",
                )
            self._client_settings = _client_settings
            self._client = bagel.Client(_client_settings)

        self._cluster = self._client.get_or_create_cluster(
            name=cluster_name,
            metadata=cluster_metadata,
        )
        self.override_relevance_score_fn = relevance_score_fn
        self._embedding_function = embedding_function

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    @xor_args(("query_texts", "query_embeddings"))
    def __query_cluster(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Query the Bagel cluster based on the provided parameters."""
        try:
            import bagel  # noqa: F401
        except ImportError:
            raise ImportError("Please install bagel `pip install bagelML`.")

        if self._embedding_function and query_embeddings is None and query_texts:
            texts = list(query_texts)
            query_embeddings = self._embedding_function.embed_documents(texts)
            query_texts = None

        return self._cluster.find(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts along with their corresponding embeddings and optional
        metadata to the Bagel cluster.

        Args:
            texts (Iterable[str]): Texts to be added.
            embeddings (Optional[List[float]]): List of embeddingvectors
            metadatas (Optional[List[dict]]): Optional list of metadatas.
            ids (Optional[List[str]]): List of unique ID for the texts.

        Returns:
            List[str]: List of unique ID representing the added texts.
        """
        # creating unique ids if None
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        texts = list(texts)
        if self._embedding_function and embeddings is None and texts:
            embeddings = self._embedding_function.embed_documents(texts)
        if metadatas:
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, metadata in enumerate(metadatas):
                if metadata:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                self._cluster.upsert(
                    embeddings=embeddings_with_metadatas,
                    metadatas=metadatas,
                    documents=texts_with_metadatas,
                    ids=ids_with_metadata,
                )
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._cluster.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            metadatas = [{}] * len(texts)
            self._cluster.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Run a similarity search with Bagel.

        Args:
            query (str): The query text to search for similar documents/texts.
            k (int): The number of results to return.
            where (Optional[Dict[str, str]]): Metadata filters to narrow down.

        Returns:
            List[Document]: List of documents objects representing
            the documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, where=where)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Run a similarity search with Bagel and return documents with their
        corresponding similarity scores.

        Args:
            query (str): The query text to search for similar documents.
            k (int): The number of results to return.
            where (Optional[Dict[str, str]]): Filter using metadata.

        Returns:
            List[Tuple[Document, float]]: List of tuples, each containing a
            Document object representing a similar document and its
            corresponding similarity score.

        """
        results = self.__query_cluster(query_texts=[query], n_results=k, where=where)
        return _results_to_docs_and_scores(results)

    @classmethod
    def from_texts(
        cls: Type[Bagel],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        cluster_name: str = _LANGCHAIN_DEFAULT_CLUSTER_NAME,
        client_settings: Optional[bagel.config.Settings] = None,
        cluster_metadata: Optional[Dict] = None,
        client: Optional[bagel.Client] = None,
        text_embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> Bagel:
        """
        Create and initialize a Bagel instance from list of texts.

        Args:
            texts (List[str]): List of text content to be added.
            cluster_name (str): The name of the Bagel cluster.
            client_settings (Optional[bagel.config.Settings]): Client settings.
            cluster_metadata (Optional[Dict]): Metadata of the cluster.
            embeddings (Optional[Embeddings]): List of embedding.
            metadatas (Optional[List[dict]]): List of metadata.
            ids (Optional[List[str]]): List of unique ID. Defaults to None.
            client (Optional[bagel.Client]): Bagel client instance.

        Returns:
            Bagel: Bagel vectorstore.
        """
        bagel_cluster = cls(
            cluster_name=cluster_name,
            embedding_function=embedding,
            client_settings=client_settings,
            client=client,
            cluster_metadata=cluster_metadata,
            **kwargs,
        )
        _ = bagel_cluster.add_texts(
            texts=texts, embeddings=text_embeddings, metadatas=metadatas, ids=ids
        )
        return bagel_cluster

    def delete_cluster(self) -> None:
        """Delete the cluster."""
        self._client.delete_cluster(self._cluster.name)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        query_embeddings: List[float],
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return docs most similar to embedding vector and similarity score.
        """
        results = self.__query_cluster(
            query_embeddings=query_embeddings, n_results=k, where=where
        )
        return _results_to_docs_and_scores(results)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector."""
        results = self.__query_cluster(
            query_embeddings=embedding, n_results=k, where=where
        )
        return _results_to_docs(results)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Select and return the appropriate relevance score function based
        on the distance metric used in the Bagel cluster.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        distance = "l2"
        distance_key = "hnsw:space"
        metadata = self._cluster.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]

        if distance == "cosine":
            return self._cosine_relevance_score_fn
        elif distance == "l2":
            return self._euclidean_relevance_score_fn
        elif distance == "ip":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function for distance"
                f" metric of type: {distance}. Consider providing"
                " relevance_score_fn to Bagel constructor."
            )

    @classmethod
    def from_documents(
        cls: Type[Bagel],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        cluster_name: str = _LANGCHAIN_DEFAULT_CLUSTER_NAME,
        client_settings: Optional[bagel.config.Settings] = None,
        client: Optional[bagel.Client] = None,
        cluster_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Bagel:
        """
        Create a Bagel vectorstore from a list of documents.

        Args:
            documents (List[Document]): List of Document objects to add to the
                                        Bagel vectorstore.
            embedding (Optional[List[float]]): List of embedding.
            ids (Optional[List[str]]): List of IDs. Defaults to None.
            cluster_name (str): The name of the Bagel cluster.
            client_settings (Optional[bagel.config.Settings]): Client settings.
            client (Optional[bagel.Client]): Bagel client instance.
            cluster_metadata (Optional[Dict]): Metadata associated with the
                                               Bagel cluster. Defaults to None.

        Returns:
            Bagel: Bagel vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            cluster_name=cluster_name,
            client_settings=client_settings,
            client=client,
            cluster_metadata=cluster_metadata,
            **kwargs,
        )

    def update_document(self, document_id: str, document: Document) -> None:
        """Update a document in the cluster.

        Args:
            document_id (str): ID of the document to update.
            document (Document): Document to update.
        """
        text = document.page_content
        metadata = document.metadata
        self._cluster.update(
            ids=[document_id],
            documents=[text],
            metadatas=[metadata],
        )

    def get(
        self,
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Gets the collection."""
        kwargs = {
            "ids": ids,
            "where": where,
            "limit": limit,
            "offset": offset,
            "where_document": where_document,
        }

        if include is not None:
            kwargs["include"] = include

        return self._cluster.get(**kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """
        Delete by IDs.

        Args:
            ids: List of ids to delete.
        """
        self._cluster.delete(ids=ids)
