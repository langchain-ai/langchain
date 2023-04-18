from itertools import repeat
from typing import Any, Iterable, List, Optional, Tuple, Type, Union

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class SupabaseVectorStore(VectorStore):
    """VectorStore for a Supabase postgres database. Assumes you have the `pgvector`
    extension installed and a `match_documents` (or similar) function. For more details:
    https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

    You can implement your own `match_documents` function in order to limit the search space
    to a subset of documents based on your own authorization or business logic.

    Note that the Supabase Python client does not yet support async operations.
    """

    _client: Any
    # This is the embedding function. Don't confuse with the embedding vectors.
    # We should perhaps rename the underlying Embedding base class to EmbeddingFunction or something
    _embedding: Embeddings
    table_name: str
    query_name: str

    def __init__(
        self,
        client: Any,
        embedding: Embeddings,
        table_name: str,
        query_name: Union[str, None] = None,
    ) -> None:
        """Initialize with supabase client."""
        try:
            import supabase
        except ImportError:
            raise ValueError(
                "Could not import supabase python package. "
                "Please install it with `pip install supabase`."
            )

        if not isinstance(client, supabase.client.Client):
            raise ValueError("client should be an instance of supabase.client.Client")

        self._client = client
        self._embedding: Embeddings = embedding
        self.table_name = table_name or "documents"
        self.query_name = query_name or "match_documents"

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict[Any, Any]]] = None,
    ) -> List[str]:
        docs = self._texts_to_documents(texts, metadatas)

        vectors = self._embedding.embed_documents(list(texts))
        return self.add_vectors(vectors, docs)

    @classmethod
    def from_texts(
        cls: Type["SupabaseVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict[Any, Any]]],
        client: Any,
        table_name: str,
        query_name: Union[str, None] = None,
        **kwargs: Any,
    ) -> "SupabaseVectorStore":
        """Return VectorStore initialized from texts and embeddings."""

        embeddings = embedding.embed_documents(texts)
        docs = cls._texts_to_documents(texts, metadatas)
        _ids = cls._add_vectors(client, table_name, embeddings, docs)

        return cls(
            client=client,
            embedding=embedding,
            table_name=table_name,
            query_name=query_name,
        )

    @classmethod
    def from_documents(
        cls: Type["SupabaseVectorStore"],
        documents: List[Document],
        embedding: Embeddings,
        client: Any,
        table_name: str,
        query_name: Union[str, None] = None,
    ) -> "SupabaseVectorStore":
        """Return VectorStore initialized from Documents."""

        texts = [doc.page_content for doc in documents]
        embeddings = embedding.embed_documents(texts)
        _ids = cls._add_vectors(client, table_name, embeddings, documents)

        return cls(
            client=client,
            embedding=embedding,
            table_name=table_name,
            query_name=query_name,
        )

    def add_vectors(
        self, vectors: List[List[float]], documents: List[Document]
    ) -> List[str]:
        return self._add_vectors(self._client, self.table_name, vectors, documents)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        vectors = self._embedding.embed_documents([query])
        return self.similarity_search_by_vector(vectors[0], k)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        result = self.similarity_search_by_vector_with_relevance_scores(embedding, k)

        documents = [doc for doc, _ in result]

        return documents

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        vectors = self._embedding.embed_documents([query])
        return self.similarity_search_by_vector_with_relevance_scores(vectors[0], k)

    def similarity_search_by_vector_with_relevance_scores(
        self, query: List[float], k: int
    ) -> List[Tuple[Document, float]]:
        match_documents_params = dict(query_embedding=query, match_count=k)
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

    @staticmethod
    def _texts_to_documents(
        texts: Iterable[str], metadatas: Optional[Iterable[dict[Any, Any]]] = None
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
        client: Any,
        table_name: str,
        vectors: List[List[float]],
        documents: List[Document],
    ) -> List[str]:
        """Add vectors to Supabase table."""
        try:
            import supabase
        except ImportError:
            raise ValueError(
                "Could not import supabase python package. "
                "Please install it with `pip install supabase`."
            )

        if not isinstance(client, supabase.client.Client):
            raise ValueError("client should be an instance of supabase.client.Client")

        rows: List[dict[str, Any]] = [
            {
                "content": documents[idx].page_content,
                "embedding": embedding,
                "metadata": documents[idx].metadata,  # type: ignore
            }
            for idx, embedding in enumerate(vectors)
        ]

        # According to the SupabaseVectorStore JS implementation, the best chunk size is 500
        chunk_size = 500
        id_list: List[str] = []
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]

            result = client.from_(table_name).insert(chunk).execute()  # type: ignore

            if len(result.data) == 0:
                raise Exception("Error inserting: No rows added")

            # VectorStore.add_vectors returns ids as strings
            ids = [str(i.get("id")) for i in result.data if i.get("id")]

            id_list.extend(ids)

        return id_list
