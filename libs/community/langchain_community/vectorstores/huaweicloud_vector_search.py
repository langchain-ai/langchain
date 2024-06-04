import logging
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
    Union,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class CSSVectorStore(VectorStore):
    """`Huawei Elasticsearch` vector store.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import CSSVectorStore
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vectorstore = CSSVectorStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                css_url="http://localhost:9200"
            )

    Args:
        index_name: Name of the Elasticsearch index to create.
        css_url: URL of the Huawei Elasticsearch instance to connect to.
        user: Username to use when connecting to Elasticsearch.
        password: Password to use when connecting to Elasticsearch.

    """

    def __init__(
        self,
        index_name: str,
        css_url: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        **kwargs: Optional[dict],
    ) -> None:
        self.embedding = embedding
        self.index_name = index_name
        self.query_field = kwargs.get("query_field", "text")
        self.vector_query_field = kwargs.get("vector_query_field", "vector")
        self.indexing = kwargs.get("indexing", True)
        self.index_type = kwargs.get("index_type", "GRAPH")
        self.metric_type = kwargs.get("metric_type", "cosine")
        self.index_params = kwargs.get("index_params") or {}

        if css_url is not None:
            self.client = CSSVectorStore.css_client(
                css_url=css_url, username=user, password=password
            )
        else:
            raise ValueError("""Please specified a css connection url.""")

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    @staticmethod
    def css_client(
        *,
        css_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> "Elasticsearch":
        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        connection_params: Dict[str, Any] = {}

        connection_params["hosts"] = [css_url]
        if username and password:
            connection_params["basic_auth"] = (username, password)

        es_client = elasticsearch.Elasticsearch(**connection_params)
        try:
            es_client.info()
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise e
        return es_client

    def _create_index_if_not_exists(self, dimension: Optional[int] = None) -> None:
        """Create the index if it doesn't already exist.

        Args:
            dimension: Length of the embedding vectors.
        """

        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists. Skipping creation.")

        else:
            if dimension is None:
                raise ValueError(
                    "Cannot create index without specifying dimension "
                    + "when the index doesn't already exist. "
                )

            indexMapping = self._index_mapping(dimension=dimension)

            logger.debug(
                f"Creating index {self.index_name} with mappings {indexMapping}"
            )

            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {"index": {"vector": True}},
                    "mappings": {"properties": indexMapping},
                },
            )

    def _index_mapping(self, dimension: Union[int, None]) -> Dict:
        """
        Executes when the index is created.

        Args:
            dimension: Numeric length of the embedding vectors,
                        or None if not using vector-based query.

        Returns:
            Dict: The Elasticsearch settings and mappings for the strategy.
        """
        if self.indexing:
            return {
                self.vector_query_field: {
                    "type": "vector",
                    "dimension": dimension,
                    "indexing": True,
                    "algorithm": self.index_type,
                    "metric": self.metric_type,
                    **self.index_params,
                }
            }
        else:
            return {self.vector_query_field: {"type": "vector", "dimension": dimension}}

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents from the index.

        Args:
            ids: List of ids of documents to delete
        """
        try:
            from elasticsearch.helpers import BulkIndexError, bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        body = []

        if ids is None:
            raise ValueError("ids must be provided.")

        for _id in ids:
            body.append({"_op_type": "delete", "_index": self.index_name, "_id": _id})

        if len(body) > 0:
            try:
                bulk(
                    self.client,
                    body,
                    refresh=kwargs.get("refresh_indices", True),
                    ignore_status=404,
                )
                logger.debug(f"Deleted {len(body)} texts from index")
                return True
            except BulkIndexError as e:
                logger.error(f"Error deleting texts: {e}")
                raise e
        else:
            logger.info("No documents to delete")
            return False

    def _query_body(
        self,
        k,
        query_vector: Union[List[float], None],
        filter: Optional[dict] = None,
    ) -> Dict:
        if self.indexing:
            return {
                "size": k,
                "query": {
                    "bool": {
                        "filter": filter,
                        "must": [
                            {
                                "vector": {
                                    self.vector_query_field: {
                                        "vector": query_vector,
                                        "topk": k,
                                    }
                                }
                            }
                        ],
                    }
                },
            }
        else:
            return {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": filter}},
                        "script": {
                            "source": "vector_score",
                            "lang": "vector",
                            "params": {
                                "field": self.vector_query_field,
                                "vector": query_vector,
                                "metric": self.metric_type,
                            },
                        },
                    }
                },
            }

    def _search(
        self,
        k: int = 4,
        query: Optional[str] = None,
        query_vector: Union[List[float], None] = None,
        filter: Optional[dict] = None,
        custom_query: Optional[Callable[[Dict, Union[str, None]], Dict]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return searched documents result from CSS

        Args:
            query: Text to look up documents similar to.
            query_vector: Embedding to look up documents similar to.
            filter: Array of Huawei ElasticSearch filter clauses to apply to the query.
            custom_query: Function to modify the query body before it is sent to CSS.

        Returns:
            List of Documents most similar to the query and score for each
        """

        if self.embedding and query is not None:
            query_vector = self.embedding.embed_query(query)

        query_body = self._query_body(k=k, query_vector=query_vector, filter=filter)

        if custom_query is not None:
            query_body = custom_query(query_body, query)
            logger.debug(f"Calling custom_query, Query body now: {query_body}")

        logger.debug(f"Query body: {query_body}")

        # Perform the kNN search on the index and return the results.
        response = self.client.search(index=self.index_name, body=query_body)
        logger.debug(f"response={response}")

        hits = [hit for hit in response["hits"]["hits"]]
        docs_and_scores = [
            (
                Document(
                    page_content=hit["_source"][self.query_field],
                    metadata=hit["_source"]["metadata"],
                ),
                hit["_score"],
            )
            for hit in hits
        ]

        return docs_and_scores

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the query,
            in descending order of similarity.
        """

        results = self.similarity_search_with_score(
            query=query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self, query: str, k: int, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            size: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the query and score for each
        """

        return self._search(k=k, query=query, filter=filter, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "CSSVectorStore":
        """Construct CSSVectorStore wrapper from documents.

        Args:
            documents: List of documents to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
                      Do not provide if using a strategy
                      that doesn't require inference.
            kwargs: create index key words arguments
        """

        vectorStore = CSSVectorStore._css_vector_store(embedding=embedding, **kwargs)
        # Encode the provided texts and add them to the newly created index.
        vectorStore.add_documents(documents)

        return vectorStore

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "CSSVectorStore":
        """Construct CSSVectorStore wrapper from raw documents.

        Args:
            texts: List of texts to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
            metadatas: Optional list of metadatas associated with the texts.
            index_name: Name of the Elasticsearch index to create.
            kwargs: create index key words arguments
        """

        vectorStore = CSSVectorStore._css_vector_store(embedding=embedding, **kwargs)

        # Encode the provided texts and add them to the newly created index.
        vectorStore.add_texts(texts, metadatas=metadatas, **kwargs)

        return vectorStore

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        try:
            from elasticsearch.helpers import BulkIndexError, bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        embeddings = []
        create_index_if_not_exists = kwargs.get("create_index_if_not_exists", True)
        ids = kwargs.get("ids", [str(uuid.uuid4()) for _ in texts])
        refresh_indices = kwargs.get("refresh_indices", True)
        requests = []

        if self.embedding is not None:
            embeddings = self.embedding.embed_documents(list(texts))
            dims_length = len(embeddings[0])

            if create_index_if_not_exists:
                self._create_index_if_not_exists(dims_length=dims_length)

            for i, (text, vector) in enumerate(zip(texts, embeddings)):
                metadata = metadatas[i] if metadatas else {}

                requests.append(
                    {
                        "_op_type": "index",
                        "_index": self.index_name,
                        self.query_field: text,
                        self.vector_query_field: vector,
                        "metadata": metadata,
                        "_id": ids[i],
                    }
                )

        else:
            if create_index_if_not_exists:
                self._create_index_if_not_exists()

            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else {}

                requests.append(
                    {
                        "_op_type": "index",
                        "_index": self.index_name,
                        self.query_field: text,
                        "metadata": metadata,
                        "_id": ids[i],
                    }
                )

        if len(requests) > 0:
            try:
                success, failed = bulk(
                    self.client, requests, stats_only=True, refresh=refresh_indices
                )
                logger.debug(
                    f"Added {success} and failed to add {failed} texts to index"
                )

                logger.debug(f"added texts {ids} to index")
                return ids
            except BulkIndexError as e:
                logger.error(f"Error adding texts: {e}")
                firstError = e.errors[0].get("index", {}).get("error", {})
                logger.error(f"First error reason: {firstError.get('reason')}")
                raise e

        else:
            logger.debug("No texts to add to index")
            return []

    @staticmethod
    def _css_vector_store(
        embedding: Optional[Embeddings] = None, **kwargs: Any
    ) -> "CSSVectorStore":
        index_name = kwargs.get("index_name")

        if index_name is None:
            raise ValueError("Please provide an index_name.")

        css_url = kwargs.get("css_url")
        if css_url is None:
            raise ValueError("Please provided a valid css connection url")

        return CSSVectorStore(embedding=embedding, **kwargs)
