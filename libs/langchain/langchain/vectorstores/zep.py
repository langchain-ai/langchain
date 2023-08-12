import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from zep_python.document import Document as ZepDocument
    from zep_python.document import DocumentCollection


class ZepVectorStore(VectorStore):
    """A vectorstore that uses Zep as the backend.

    Search scores are calculated using cosine similarity normalized to [0, 1].

    Args:
        collection: A Zep DocumentCollection.
        texts: Optional list of texts to add to the vectorstore.
        metadata: Optional list of metadata associated with the texts.
        embedding: Optional embedding function to use to embed the texts.
        **kwargs: vectorstore specific parameters
    """

    def __init__(
        self,
        collection: "DocumentCollection",
        texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import zep_python
            from zep_python.document import DocumentCollection
        except ImportError:
            raise ValueError(
                "Could not import zep-python python package. "
                "Please install it with `pip install zep-python`."
            )
        if not isinstance(collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        self._collection: Optional[DocumentCollection] = collection
        self._texts: Optional[List[str]] = texts
        self._embedding: Optional[Embeddings] = embedding

        if self._texts is not None:
            self.add_texts(self._texts, metadatas=metadata, **kwargs)

    def _generate_documents_to_add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List["ZepDocument"]:
        from zep_python.document import Document as ZepDocument

        if (
            self._collection
            and self._collection.is_auto_embedded
            and self._embedding is not None
        ):
            warnings.warn(
                """The collection is set to auto-embed and an embedding 
            function is present. Ignoring the embedding function.""",
                stacklevel=2,
            )
            self._embedding = None

        embeddings = None
        if self._embedding is not None:
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
        from zep_python.document import DocumentCollection

        if not isinstance(self._collection, DocumentCollection):
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
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        documents = self._generate_documents_to_add(texts, metadatas, document_ids)
        uuids = await self._collection.aadd_documents(documents)

        return uuids

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents List[Document]: Documents to add to the vectorstore.

        Returns:
            List[str]: List of UUIDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents List[Document]: Documents to add to the vectorstore.

        Returns:
            List[str]: List of UUIDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas, **kwargs)

    def search(
        self,
        query: str,
        search_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query using specified search type."""
        if search_type == "similarity":
            return self.similarity_search(query, k, metadata, **kwargs)
        elif search_type == "mmr":
            return self.max_marginal_relevance_search(
                query, k, metadata=metadata, **kwargs
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
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query using specified search type."""
        if search_type == "similarity":
            return await self.asimilarity_search(query, k, metadata, **kwargs)
        elif search_type == "mmr":
            return await self.amax_marginal_relevance_search(
                query, k, metadata=metadata, **kwargs
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
            query, k, metadata=metadata, **kwargs
        )
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""

        return self._similarity_search_with_relevance_scores(
            query, k, metadata=metadata, **kwargs
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

        if not isinstance(self._collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if self._embedding:
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

        if not isinstance(self._collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if self._embedding:
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
        if not isinstance(self._collection, DocumentCollection):
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
        if not isinstance(self._collection, DocumentCollection):
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

    def _max_marginal_relevance_selection(
        self,
        query_vector: List[float],
        results: List["ZepDocument"],
        k: int = 4,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        mmr_selected = maximal_marginal_relevance(
            np.array([query_vector], dtype=np.float32),
            [d.embedding for d in results],
            k=k,
            lambda_mult=lambda_mult,
        )
        selected = [results[i] for i in mmr_selected]
        return [Document(page_content=d.content, metadata=d.metadata) for d in selected]

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
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            metadata: Optional, metadata to filter the resulting set of retrieved docs
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        if not isinstance(self._collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if self._embedding:
            query_vector = self._embedding.embed_query(query)
            results = self._collection.search(
                embedding=query_vector, limit=k, metadata=metadata, **kwargs
            )
        else:
            results, query_vector = self._collection.search_return_query_vector(
                query, limit=k, metadata=metadata, **kwargs
            )

        return self._max_marginal_relevance_selection(
            query_vector, results, k=k, lambda_mult=lambda_mult
        )

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

        if not isinstance(self._collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        if self._embedding:
            query_vector = self._embedding.embed_query(query)
            results = await self._collection.asearch(
                embedding=query_vector, limit=k, metadata=metadata, **kwargs
            )
        else:
            results, query_vector = await self._collection.asearch_return_query_vector(
                query, limit=k, metadata=metadata, **kwargs
            )

        return self._max_marginal_relevance_selection(
            query_vector, results, k=k, lambda_mult=lambda_mult
        )

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
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            metadata: Optional, metadata to filter the resulting set of retrieved docs
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        results = self._collection.search(
            embedding=embedding, limit=k, metadata=metadata, **kwargs
        )

        return self._max_marginal_relevance_selection(
            embedding, results, k=k, lambda_mult=lambda_mult
        )

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
        if not isinstance(self._collection, DocumentCollection):
            raise ValueError(
                "collection should be an instance of a Zep DocumentCollection"
            )

        results = await self._collection.asearch(
            embedding=embedding, limit=k, metadata=metadata, **kwargs
        )

        return self._max_marginal_relevance_selection(
            embedding, results, k=k, lambda_mult=lambda_mult
        )

    @classmethod
    def from_texts(  # type: ignore
        cls: Type["ZepVectorStore"],
        texts: List[str],
        collection: "DocumentCollection",
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "ZepVectorStore":
        """Return VectorStore initialized from texts and embeddings."""
        return cls(
            collection=collection, texts=texts, embedding=embedding, metadata=metadatas
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

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
