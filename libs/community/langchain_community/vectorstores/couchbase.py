from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Default batch size
DEFAULT_BATCH_SIZE = 100

if TYPE_CHECKING:
    from couchbase.cluster import Cluster


class CouchbaseVectorStore(VectorStore):
    """Wrapper around Couchbase for vector-store workloads.

    To use it, you need a recent installation of the `couchbase` library
    and a Couchbase database with support for Vector Search.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import CouchbaseVectorStore
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            from couchbase import Cluster
            from couchbase.auth import PasswordAuthenticator
            from couchbase.options import ClusterOptions
            from datetime import timedelta

            auth = PasswordAuthenticator(username, password)
            options = ClusterOptions(auth)
            connect_string = "couchbases://localhost"
            cluster = Cluster(connect_string, options)

            # Wait until the cluster is ready for use.
            cluster.wait_until_ready(timedelta(seconds=5))

            embeddings = OpenAIEmbeddings()

            vectorstore = CouchbaseVectorStore(
                client=cluster,
                bucket_name="",
                scope_name="",
                collection_name="",
                embedding=embeddings,
                index_name="vector-index",
            )

            vectorstore.add_texts(["hello", "world"])
            results = vectorstore.similarity_search("ola", k=1)

    Constructor Args:
        cluster (Cluster): couchbase cluster object with active connection.
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
        cluster: Cluster,
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        embedding: Embeddings,
        index_name: str,
        *,
        text_key: Optional[str] = "text",
        embedding_key: Optional[str] = None,
    ) -> None:
        try:
            from couchbase.cluster import Cluster
        except ImportError as e:
            raise ImportError(
                "Could not import couchbase python package. "
                "Please install couchbase SDK  with `pip install couchbase`."
            ) from e

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

        self._bucket_name = bucket_name
        self._scope_name = scope_name
        self._collection_name = collection_name
        self._embedding_function = embedding
        self._text_key = text_key
        self._index_name = index_name

        if not isinstance(cluster, Cluster):
            raise ValueError(
                f"cluster should be an instance of couchbase.Cluster, "
                f"got {type(cluster)}"
            )

        self._cluster = cluster

        # Wait until the cluster is ready for use.
        # self._cluster.wait_until_ready(timedelta(seconds=5))

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
        import couchbase.search as search
        from couchbase.options import SearchOptions
        from couchbase.vector_search import VectorQuery, VectorSearch

        search_req = search.SearchRequest.create(
            search.MatchAllQuery()
        ).with_vector_search(
            VectorSearch.from_vector_query(
                VectorQuery(
                    field_name=self._embedding_key,
                    vector=embedding,
                    num_candidates=k * 10,
                )
            )
        )

        search_iter = self._scope.search(
            self._index_name,
            search_req,
            SearchOptions(limit=k, fields=[self._text_key, "metadata"]),
        )

        docs_with_score = []

        for row in search_iter.rows():
            text = row.fields.pop(self._text_key)
            metadata = row.fields
            score = row.score
            doc = Document(page_content=text, metadata=metadata)
            docs_with_score.append((doc, score))
            print(f"row: {row}")
        # db_host = self._connection_string.split("//")[-1].strip("/")

        # search_query = {
        #     "fields": [self._text_key, "metadata"],
        #     "sort": ["-_score"],
        #     "limit": k,
        #     "query": {"match_none": {}},
        #     "knn": [{"k": k * 10, "field": self._embedding_key, "vector": embedding}],
        # }

        # search_result = httpx.post(
        #     f"http://{db_host}:8094/api/bucket/{self._bucket_name}/scope/{self._scope_name}/index/{self._index_name}/query",
        #     json=search_query,
        #     auth=(self._db_username, self._db_password),
        #     headers={"Content-Type": "application/json"},
        # )

        # if search_result.status_code == 200:
        #     response_json = search_result.json()
        #     results = response_json["hits"]

        #     for result in results:
        #         text = result["fields"].pop(self._text_key)
        #         score = result["score"]
        #         doc = Document(page_content=text, metadata=result["fields"])
        #         docs_with_score.append((doc, score))
        # else:
        #     raise ValueError(
        #         f"Request failed with status code {search_result.status_code}"
        #         " and error message: {search_result.text}"
        #     )

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
        cluster: Cluster,
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

            from couchbase import Cluster
            from couchbase.auth import PasswordAuthenticator
            from couchbase.options import ClusterOptions
            from datetime import timedelta

            auth = PasswordAuthenticator(username, password)
            options = ClusterOptions(auth)
            connect_string = "couchbases://localhost"
            cluster = Cluster(connect_string, options)

            # Wait until the cluster is ready for use.
            cluster.wait_until_ready(timedelta(seconds=5))

            embeddings = OpenAIEmbeddings()

            texts = ["hello", "world"]

            vectorstore = CouchbaseVectorStore.from_texts(
                texts,
                embedding=embeddings,
                cluster=cluster,
                bucket_name="",
                scope_name="",
                collection_name="",
                index_name="vector-index",
            )

        Args:
            texts (List[str]): list of texts to add to the vector store.
            embedding (Embeddings): embedding function to use.
            cluster (Cluster): couchbase cluster object with active connection.
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
        if not bucket_name:
            raise ValueError("bucket_name must be provided.")

        if not scope_name:
            raise ValueError("scope_name must be provided.")

        if not collection_name:
            raise ValueError("collection_name must be provided.")

        if not index_name:
            raise ValueError("index_name must be provided.")

        vector_store = cls(
            cluster,
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
