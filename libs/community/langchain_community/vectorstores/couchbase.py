from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Default batch size
DEFAULT_BATCH_SIZE = 100


class CouchbaseVectorStore(VectorStore):
    """Wrapper around Couchbase for vector-store workloads.

    To use it, you need a recent installation of the `couchbase` library
    and a Couchbase database with support for Vector Search.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import CouchbaseVectorStore
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()

            vectorstore = CouchbaseVectorStore(
                connection_string="couchbase://",
                db_username="",
                db_password="",
                bucket_name="",
                scope_name="",
                collection_name="",
                embedding=embeddings,
                index_name="vector-index",
            )

            vectorstore.add_texts(["hello", "world"])
            results = vectorstore.similarity_search("ola", k=1)

    Constructor Args:
        connection_string (str): connection string for couchbase cluster.
        db_username (str): database user with read/write access to bucket.
        db_password (str): database password with read/write access to bucket.
        bucket_name (str): name of bucket to store documents in.
        scope_name (str): name of scope to store documents in.
        collection_name (str): name of collection to store documents in.
        embedding (Embeddings): embedding function to use.
        index_name (str): name of the index to use.
        text_key (optional[str]): key in document to use as text.
        embedding_key (optional[str]): key in document to use for the embeddings.
    """

    def __init__(
        self,
        connection_string: str,
        db_username: str,
        db_password: str,
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        embedding: Embeddings,
        index_name: str,
        text_key: Optional[str] = "text",
        *,
        embedding_key: Optional[str] = None,
    ) -> None:
        try:
            from couchbase.auth import PasswordAuthenticator
            from couchbase.cluster import Cluster
            from couchbase.options import ClusterOptions
        except ImportError as e:
            raise ImportError(
                "Could not import couchbase python package. "
                "Please install couchbase SDK  with `pip install couchbase`."
            ) from e

        if not connection_string:
            raise ValueError("connection_string must be provided.")

        if not db_username:
            raise ValueError("db_username must be provided.")

        if not db_password:
            raise ValueError("db_password must be provided.")

        if not bucket_name:
            raise ValueError("bucket_name must be provided.")

        if not scope_name:
            raise ValueError("scope_name must be provided.")

        if not collection_name:
            raise ValueError("collection_name must be provided.")

        if not index_name:
            raise ValueError("index_name must be provided.")

        if not embedding_key:
            self._embedding_key = text_key + "_embedding"

        self._connection_string = connection_string
        self._db_username = db_username
        self._db_password = db_password
        self._bucket_name = bucket_name
        self._scope_name = scope_name
        self._collection_name = collection_name
        self._embedding_function = embedding
        self._text_key = text_key
        self._index_name = index_name

        auth = PasswordAuthenticator(
            self._db_username,
            self._db_password,
        )
        self._cluster: Cluster = Cluster(self._connection_string, ClusterOptions(auth))
        # Wait until the cluster is ready for use.
        self._cluster.wait_until_ready(timedelta(seconds=5))

        self._bucket = self._cluster.bucket(self._bucket_name)
        self._scope = self._bucket.scope(self._scope_name)
        self._collection = self._scope.collection(self._collection_name)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through the embeddings and persist in vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings to add to the vectorstore.
            metadatas (Optional[List[Dict]]): Optional list of metadatas associated
                with the texts.
            ids (Optional[List[str]]): Optional list of ids associated with the texts.
                IDs have to be unique strings across the collection.
                If it is not specified uuids are generated and used as ids.
            batch_size (Optional[int]): Optional batch size for bulk insertions.
                Default:100

        Returns:
            List[str]:List of ids from adding the texts into the vectorstore.
        """
        from couchbase.exceptions import DocumentExistsException

        batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        doc_ids = []

        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        embedded_texts = self._embedding_function.embed_documents(texts)

        documents_to_insert = [
            {
                id: {
                    self._text_key: text,
                    self._embedding_key: vector,
                    "metadata": metadata,
                }
                for id, text, vector, metadata in zip(
                    ids, texts, embedded_texts, metadatas
                )
            }
        ]

        for i in range(0, len(documents_to_insert), batch_size):
            batch = documents_to_insert[i : i + batch_size]
            try:
                result = self._collection.upsert_multi(batch[0])
                if result.all_ok:
                    doc_ids.extend(batch[0].keys())
            except DocumentExistsException as e:
                raise ValueError(f"Document already exists: {e}")

        return doc_ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the vector store by ids.

        Args:
            ids (List[str]): List of IDs of the documents to delete.
            batch_size (Optional[int]): Optional batch size for bulk deletions.

        Returns:
            bool: True if all the documents were deleted successfully, False otherwise.

        """
        from couchbase.exceptions import DocumentNotFoundException

        if ids is None:
            raise ValueError("No document ids provided to delete.")

        batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        deletion_status = True

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            try:
                result = self._collection.remove_multi(batch)
            except DocumentNotFoundException as e:
                deletion_status = False
                raise ValueError(f"Document not found: {e}")

            deletion_status &= result.all_ok

        return deletion_status

    @property
    def embeddings(self) -> Embeddings:
        """Return the query embedding object."""
        return self._embedding_function

    # def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
    #     """Add documents to the vector store."""
    #     return super().add_documents(documents, **kwargs)
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector with their scores.
        Args:
            embedding (List[float]): Embedding vector to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.

        Returns:
            List of (Document, score) that are the most similar to the query vector.
        """
        db_host = self._connection_string.split("//")[-1].strip("/")

        search_query = {
            "fields": [self._text_key, "metadata"],
            "sort": ["-_score"],
            "limit": k,
            "query": {"match_none": {}},
            "knn": [{"k": k * 10, "field": self._embedding_key, "vector": embedding}],
        }

        search_result = httpx.post(
            f"http://{db_host}:8094/api/bucket/{self._bucket_name}/scope/{self._scope_name}/index/{self._index_name}/query",
            json=search_query,
            auth=(self._db_username, self._db_password),
            headers={"Content-Type": "application/json"},
        )

        if search_result.status_code == 200:
            response_json = search_result.json()
            results = response_json["hits"]
            docs_with_score = []
            for result in results:
                text = result["fields"].pop(self._text_key)
                score = result["score"]
                doc = Document(page_content=text, metadata=result["fields"])
                docs_with_score.append((doc, score))
        else:
            raise ValueError(
                f"Request failed with status code {search_result.status_code}"
                " and error message: {search_result.text}"
            )

        return docs_with_score

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return documents that are most similar to the query.
        Args:
            query (str): Query to look up for similar documents
            k (int): Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query."""
        docs_with_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with score."
        Args:
            query (str): Query to look up for similar documents
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score) that are most similar to the query.
        """
        query_embedding = self.embeddings.embed_query(query)
        docs_with_score = self.similarity_search_with_score_by_vector(
            query_embedding, k
        )
        return docs_with_score

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents that are most similar to the vector embedding.
        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        docs_with_score = self.similarity_search_with_score_by_vector(embedding, k)
        return [doc for doc, _ in docs_with_score]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        connection_string: str,
        db_username: str,
        db_password: str,
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        index_name: str,
        text_key: Optional[str] = "text",
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embedding_key: Optional[str] = None,
    ) -> CouchbaseVectorStore:
        """Construct a Couchbase vector store from a list of texts.

        Example:
            .. code-block:: python

            from langchain_community.vectorstores import CouchbaseVectorStore
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()

            texts = ["hello", "world"]

            vectorstore = CouchbaseVectorStore.from_texts(
                texts,
                embedding=embeddings,
                connection_string="couchbase://",
                db_username="",
                db_password="",
                bucket_name="",
                scope_name="",
                collection_name="",
                index_name="vector-index",
            )

        Args:
            texts (List[str]): list of texts to add to the vector store.
            embedding (Embeddings): embedding function to use.
            connection_string (str): connection string for Couchbase cluster.
            db_username (str): database user with read/write access to bucket.
            db_password (str): database password with read/write access to bucket.
            bucket_name (str): name of bucket to store documents in.
            scope_name (str): name of scope to store documents in.
            collection_name (str): name of collection to store documents in.
            index_name (str): name of the index to use.
            text_key (optional[str]): key in document to use as text.
            ids (optional[List[str]]): list of ids to add to documents.
            metadatas (optional[List[Dict]): list of metadatas to add to documents.
            embedding_key (optional[str]): key in document to use for the embeddings.

        Returns:
            A Couchbase vector store.

        """
        if not connection_string:
            raise ValueError("connection_string must be provided.")

        if not db_username:
            raise ValueError("db_username must be provided.")

        if not db_password:
            raise ValueError("db_password must be provided.")

        if not bucket_name:
            raise ValueError("bucket_name must be provided.")

        if not scope_name:
            raise ValueError("scope_name must be provided.")

        if not collection_name:
            raise ValueError("collection_name must be provided.")

        if not index_name:
            raise ValueError("index_name must be provided.")

        vector_store = cls(
            connection_string,
            db_username,
            db_password,
            bucket_name,
            scope_name,
            collection_name,
            embedding,
            index_name,
            text_key,
            embedding_key,
        )

        vector_store.add_texts(texts, metadatas=metadatas, ids=ids)

        return vector_store
