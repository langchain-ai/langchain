from __future__ import annotations

import logging
import uuid
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain.utils import get_from_env
from langchain.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


class DashVector(VectorStore):
    """`DashVector` vector store.

    To use, you should have the ``dashvector`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import DashVector
            from langchain.embeddings.openai import OpenAIEmbeddings
            import dashvector

            client = dashvector.Client(api_key="***")
            client.create("langchain", dimension=1024)
            collection = client.get("langchain")
            embeddings = OpenAIEmbeddings()
            vectorstore = DashVector(collection, embeddings.embed_query, "text")
    """

    def __init__(
        self,
        collection: Any,
        embedding: Embeddings,
        text_field: str,
    ):
        """Initialize with DashVector collection."""

        try:
            import dashvector
        except ImportError:
            raise ValueError(
                "Could not import dashvector python package. "
                "Please install it with `pip install dashvector`."
            )

        if not isinstance(collection, dashvector.Collection):
            raise ValueError(
                f"collection should be an instance of dashvector.Collection, "
                f"bug got {type(collection)}"
            )

        self._collection = collection
        self._embedding = embedding
        self._text_field = text_field

    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query vector, along with scores"""

        # query by vector
        ret = self._collection.query(embedding, topk=k, filter=filter)
        if not ret:
            raise ValueError(
                f"Fail to query docs by vector, error {self._collection.message}"
            )

        docs = []
        for doc in ret:
            metadata = doc.fields
            text = metadata.pop(self._text_field)
            score = doc.score
            docs.append((Document(page_content=text, metadata=metadata), score))
        return docs

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 25,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids associated with the texts.
            batch_size: Optional batch size to upsert docs.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = ids or [str(uuid.uuid4().hex) for _ in texts]
        text_list = list(texts)
        for i in range(0, len(text_list), batch_size):
            # batch end
            end = min(i + batch_size, len(text_list))

            batch_texts = text_list[i:end]
            batch_ids = ids[i:end]
            batch_embeddings = self._embedding.embed_documents(list(batch_texts))

            # batch metadatas
            if metadatas:
                batch_metadatas = metadatas[i:end]
            else:
                batch_metadatas = [{} for _ in range(i, end)]
            for metadata, text in zip(batch_metadatas, batch_texts):
                metadata[self._text_field] = text

            # batch upsert to collection
            docs = list(zip(batch_ids, batch_embeddings, batch_metadatas))
            ret = self._collection.upsert(docs)
            if not ret:
                raise ValueError(
                    f"Fail to upsert docs to dashvector vector database,"
                    f"Error: {ret.message}"
                )
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete by vector ID.

        Args:
            ids: List of ids to delete.

        Returns:
            True if deletion is successful,
            False otherwise.
        """
        return bool(self._collection.delete(ids))

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to search documents similar to.
            k: Number of documents to return. Default to 4.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.

        Returns:
            List of Documents most similar to the query text.
        """

        docs_and_scores = self.similarity_search_with_relevance_scores(query, k, filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query text , alone with relevance scores.

        Less is more similar, more is more dissimilar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.

        Returns:
            List of Tuples of (doc, similarity_score)
        """

        embedding = self._embedding.embed_query(query)
        return self._similarity_search_with_score_by_vector(
            embedding, k=k, filter=filter
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self._similarity_search_with_score_by_vector(
            embedding, k, filter
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
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
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
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
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        # query by vector
        ret = self._collection.query(
            embedding, topk=fetch_k, filter=filter, include_vector=True
        )
        if not ret:
            raise ValueError(
                f"Fail to query docs by vector, error {self._collection.message}"
            )

        candidate_embeddings = [doc.vector for doc in ret]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), candidate_embeddings, lambda_mult, k
        )

        metadatas = [ret.output[i].fields for i in mmr_selected]
        return [
            Document(page_content=metadata.pop(self._text_field), metadata=metadata)
            for metadata in metadatas
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        dashvector_api_key: Optional[str] = None,
        collection_name: str = "langchain",
        text_field: str = "text",
        batch_size: int = 25,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DashVector:
        """Return DashVector VectorStore initialized from texts and embeddings.

        This is the quick way to get started with dashvector vector store.

        Example:
            .. code-block:: python

            from langchain.vectorstores import DashVector
            from langchain.embeddings import OpenAIEmbeddings
            import dashvector

            embeddings = OpenAIEmbeddings()
            dashvector = DashVector.from_documents(
                docs,
                embeddings,
                dashvector_api_key="{DASHVECTOR_API_KEY}"
            )
        """
        try:
            import dashvector
        except ImportError:
            raise ValueError(
                "Could not import dashvector python package. "
                "Please install it with `pip install dashvector`."
            )

        dashvector_api_key = dashvector_api_key or get_from_env(
            "dashvector_api_key", "DASHVECTOR_API_KEY"
        )

        dashvector_client = dashvector.Client(api_key=dashvector_api_key)
        dashvector_client.delete(collection_name)
        collection = dashvector_client.get(collection_name)
        if not collection:
            dim = len(embedding.embed_query(texts[0]))
            # create collection if not existed
            resp = dashvector_client.create(collection_name, dimension=dim)
            if resp:
                collection = dashvector_client.get(collection_name)
            else:
                raise ValueError(
                    "Fail to create collection. " f"Error: {resp.message}."
                )

        dashvector_vector_db = cls(collection, embedding, text_field)
        dashvector_vector_db.add_texts(texts, metadatas, ids, batch_size)
        return dashvector_vector_db
