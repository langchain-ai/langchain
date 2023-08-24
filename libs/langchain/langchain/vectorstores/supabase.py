from __future__ import annotations

import uuid
from itertools import repeat
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import supabase


class SupabaseVectorStore(VectorStore):
    """`Supabase Postgres` vector store.

    It assumes you have the `pgvector`
    extension installed and a `match_documents` (or similar) function. For more details:
    https://integrations.langchain.com/vectorstores?integration_name=SupabaseVectorStore

    You can implement your own `match_documents` function in order to limit the search
    space to a subset of documents based on your own authorization or business logic.

    Note that the Supabase Python client does not yet support async operations.

    If you'd like to use `max_marginal_relevance_search`, please review the instructions
    below on modifying the `match_documents` function to return matched embeddings.


    Examples:

    .. code-block:: python

        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.schema import Document
        from langchain.vectorstores import SupabaseVectorStore
        from supabase.client import create_client

        docs = [
            Document(page_content="foo", metadata={"id": 1}),
        ]
        embeddings = OpenAIEmbeddings()
        supabase_client = create_client("my_supabase_url", "my_supabase_key")
        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents",
        )

    To load from an existing table:

    .. code-block:: python

        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.vectorstores import SupabaseVectorStore
        from supabase.client import create_client


        embeddings = OpenAIEmbeddings()
        supabase_client = create_client("my_supabase_url", "my_supabase_key")
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents",
        )

    """

    def __init__(
        self,
        client: supabase.client.Client,
        embedding: Embeddings,
        table_name: str,
        query_name: Union[str, None] = None,
    ) -> None:
        """Initialize with supabase client."""
        try:
            import supabase  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import supabase python package. "
                "Please install it with `pip install supabase`."
            )

        self._client = client
        self._embedding: Embeddings = embedding
        self.table_name = table_name or "documents"
        self.query_name = query_name or "match_documents"

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        docs = self._texts_to_documents(texts, metadatas)

        vectors = self._embedding.embed_documents(list(texts))
        return self.add_vectors(vectors, docs, ids)

    @classmethod
    def from_texts(
        cls: Type["SupabaseVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[supabase.client.Client] = None,
        table_name: Optional[str] = "documents",
        query_name: Union[str, None] = "match_documents",
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "SupabaseVectorStore":
        """Return VectorStore initialized from texts and embeddings."""

        if not client:
            raise ValueError("Supabase client is required.")

        if not table_name:
            raise ValueError("Supabase document table_name is required.")

        embeddings = embedding.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        docs = cls._texts_to_documents(texts, metadatas)
        cls._add_vectors(client, table_name, embeddings, docs, ids)

        return cls(
            client=client,
            embedding=embedding,
            table_name=table_name,
            query_name=query_name,
        )

    def add_vectors(
        self,
        vectors: List[List[float]],
        documents: List[Document],
        ids: List[str],
    ) -> List[str]:
        return self._add_vectors(self._client, self.table_name, vectors, documents, ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        vectors = self._embedding.embed_documents([query])
        return self.similarity_search_by_vector(
            vectors[0], k=k, filter=filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        result = self.similarity_search_by_vector_with_relevance_scores(
            embedding, k=k, filter=filter, **kwargs
        )

        documents = [doc for doc, _ in result]

        return documents

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        vectors = self._embedding.embed_documents([query])
        return self.similarity_search_by_vector_with_relevance_scores(
            vectors[0], k=k, filter=filter
        )

    def match_args(
        self, query: List[float], k: int, filter: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        ret = dict(query_embedding=query, match_count=k)
        if filter:
            ret["filter"] = filter
        return ret

    def similarity_search_by_vector_with_relevance_scores(
        self, query: List[float], k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        match_documents_params = self.match_args(query, k, filter)
        res = self._client.rpc(self.query_name, match_documents_params).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        return match_result

    def similarity_search_by_vector_returning_embeddings(
        self, query: List[float], k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float, np.ndarray[np.float32, Any]]]:
        match_documents_params = self.match_args(query, k, filter)
        res = self._client.rpc(self.query_name, match_documents_params).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
                # Supabase returns a vector type as its string represation (!).
                # This is a hack to convert the string to numpy array.
                np.fromstring(
                    search.get("embedding", "").strip("[]"), np.float32, sep=","
                ),
            )
            for search in res.data
            if search.get("content")
        ]

        return match_result

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

    @staticmethod
    def _add_vectors(
        client: supabase.client.Client,
        table_name: str,
        vectors: List[List[float]],
        documents: List[Document],
        ids: List[str],
    ) -> List[str]:
        """Add vectors to Supabase table."""

        rows: List[Dict[str, Any]] = [
            {
                "id": ids[idx],
                "content": documents[idx].page_content,
                "embedding": embedding,
                "metadata": documents[idx].metadata,  # type: ignore
            }
            for idx, embedding in enumerate(vectors)
        ]

        # According to the SupabaseVectorStore JS implementation, the best chunk size
        # is 500
        chunk_size = 500
        id_list: List[str] = []
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]

            result = client.from_(table_name).upsert(chunk).execute()  # type: ignore

            if len(result.data) == 0:
                raise Exception("Error inserting: No rows added")

            # VectorStore.add_vectors returns ids as strings
            ids = [str(i.get("id")) for i in result.data if i.get("id")]

            id_list.extend(ids)

        return id_list

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
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
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        result = self.similarity_search_by_vector_returning_embeddings(
            embedding, fetch_k
        )

        matched_documents = [doc_tuple[0] for doc_tuple in result]
        matched_embeddings = [doc_tuple[2] for doc_tuple in result]

        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            matched_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        filtered_documents = [matched_documents[i] for i in mmr_selected]

        return filtered_documents

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
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
        Returns:
            List of Documents selected by maximal marginal relevance.

        `max_marginal_relevance_search` requires that `query_name` returns matched
        embeddings alongside the match documents. The following function
        demonstrates how to do this:

        ```sql
        CREATE FUNCTION match_documents_embeddings(query_embedding vector(1536),
                                                   match_count int)
            RETURNS TABLE(
                id uuid,
                content text,
                metadata jsonb,
                embedding vector(1536),
                similarity float)
            LANGUAGE plpgsql
            AS $$
            # variable_conflict use_column
        BEGIN
            RETURN query
            SELECT
                id,
                content,
                metadata,
                embedding,
                1 -(docstore.embedding <=> query_embedding) AS similarity
            FROM
                docstore
            ORDER BY
                docstore.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        ```
        """
        embedding = self._embedding.embed_documents([query])
        docs = self.max_marginal_relevance_search_by_vector(
            embedding[0], k, fetch_k, lambda_mult=lambda_mult
        )
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        rows: List[Dict[str, Any]] = [
            {
                "id": id,
            }
            for id in ids
        ]

        # TODO: Check if this can be done in bulk
        for row in rows:
            self._client.from_(self.table_name).delete().eq("id", row["id"]).execute()
