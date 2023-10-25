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

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class BESVectorStore(VectorStore):
    """`Baidu Elasticsearch` vector store.

    Example:
        .. code-block:: python

            from langchain.vectorstores import BESVectorStore
            from langchain.embeddings.openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vectorstore = BESVectorStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                bes_url="http://localhost:9200"
            )

    Args:
        index_name: Name of the Elasticsearch index to create.
        bes_url: URL of the Baidu Elasticsearch instance to connect to.
        user: Username to use when connecting to Elasticsearch.
        password: Password to use when connecting to Elasticsearch.

        More information can be obtained from:
        https://cloud.baidu.com/doc/BES/s/8llyn0hh4

    """

    def __init__(
        self,
        index_name: str,
        bes_url: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        **kwargs: Optional[dict],
    ) -> None:
        self.embedding = embedding
        self.index_name = index_name
        self.query_field = kwargs.get("query_field", "text")
        self.vector_query_field = kwargs.get("vector_query_field", "vector")
        self.space_type = kwargs.get("space_type", "cosine")
        self.index_type = kwargs.get("index_type", "linear")
        self.index_params = kwargs.get("index_params") or {}

        if bes_url is not None:
            self.client = BESVectorStore.bes_client(
                bes_url=bes_url, username=user, password=password
            )
        else:
            raise ValueError("""Please specified a bes connection url.""")

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    @staticmethod
    def bes_client(
        *,
        bes_url: Optional[str] = None,
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

        connection_params["hosts"] = [bes_url]
        connection_params["basic_auth"] = (username, password)

        es_client = elasticsearch.Elasticsearch(**connection_params)
        try:
            es_client.info()
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise e
        return es_client

    def _create_index_if_not_exists(self, dims_length: Optional[int] = None) -> None:
        """Create the index if it doesn't already exist.

        Args:
            dims_length: Length of the embedding vectors.
        """

        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists. Skipping creation.")

        else:
            if dims_length is None:
                raise ValueError(
                    "Cannot create index without specifying dims_length "
                    + "when the index doesn't already exist. "
                )

            indexMapping = self._index_mapping(dims_length=dims_length)

            logger.debug(
                f"Creating index {self.index_name} with mappings {indexMapping}"
            )

            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {"properties": indexMapping},
                },
            )

    def _index_mapping(self, dims_length: Union[int, None]) -> Dict:
        """
        Executes when the index is created.

        Args:
            dims_length: Numeric length of the embedding vectors,
                        or None if not using vector-based query.
            index_params: The extra pamameters for creating index.

        Returns:
            Dict: The Elasticsearch settings and mappings for the strategy.
        """
        if "linear" == self.index_type:
            return {
                self.vector_query_field: {
                    "type": "bpack_vector",
                    "dims": dims_length,
                    "build_index": self.index_params.get("build_index", False),
                }
            }

        elif "hnsw" == self.index_type:
            return {
                self.vector_query_field: {
                    "type": "bpack_vector",
                    "dims": dims_length,
                    "index_type": "hnsw",
                    "space_type": self.space_type,
                    "parameters": {
                        "ef_construction": self.index_params.get(
                            "hnsw_ef_construction", 200
                        ),
                        "m": self.index_params.get("hnsw_m", 4),
                    },
                }
            }
        else:
            return {
                self.vector_query_field: {
                    "type": "bpack_vector",
                    "model_id": self.index_params.get("model_id", ""),
                }
            }

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
        query_vector: Union[List[float], None],
        filter: Optional[dict] = None,
        search_params: Dict = {},
    ) -> Dict:
        query_vector_body = {"vector": query_vector, "k": search_params.get("k", 2)}

        if filter is not None and len(filter) != 0:
            query_vector_body["filter"] = filter

        if "linear" == self.index_type:
            query_vector_body["linear"] = True
            query_vector_body["space_type"] = self.space_type
        else:
            query_vector_body["ef"] = search_params.get("ef", 10)

        return {
            "size": search_params.get("size", 4),
            "query": {"knn": {self.vector_query_field: query_vector_body}},
        }

    def _search(
        self,
        query: Optional[str] = None,
        query_vector: Union[List[float], None] = None,
        filter: Optional[dict] = None,
        custom_query: Optional[Callable[[Dict, Union[str, None]], Dict]] = None,
        search_params: Dict = {},
    ) -> List[Tuple[Document, float]]:
        """Return searched documents result from BES

        Args:
            query: Text to look up documents similar to.
            query_vector: Embedding to look up documents similar to.
            filter: Array of Baidu ElasticSearch filter clauses to apply to the query.
            custom_query: Function to modify the query body before it is sent to BES.

        Returns:
            List of Documents most similar to the query and score for each
        """

        if self.embedding and query is not None:
            query_vector = self.embedding.embed_query(query)

        query_body = self._query_body(
            query_vector=query_vector, filter=filter, search_params=search_params
        )

        if custom_query is not None:
            query_body = custom_query(query_body, query)
            logger.debug(f"Calling custom_query, Query body now: {query_body}")

        logger.debug(f"Query body: {query_body}")

        # Perform the kNN search on the BES index and return the results.
        response = self.client.search(index=self.index_name, **query_body)
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
        search_params = kwargs.get("search_params") or {}

        if len(search_params) == 0 or search_params.get("size") is None:
            search_params["size"] = k

        return self._search(query=query, filter=filter, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "BESVectorStore":
        """Construct BESVectorStore wrapper from documents.

        Args:
            documents: List of documents to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
                      Do not provide if using a strategy
                      that doesn't require inference.
            kwargs: create index key words arguments
        """

        vectorStore = BESVectorStore._bes_vector_store(embedding=embedding, **kwargs)
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
    ) -> "BESVectorStore":
        """Construct BESVectorStore wrapper from raw documents.

        Args:
            texts: List of texts to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
            metadatas: Optional list of metadatas associated with the texts.
            index_name: Name of the Elasticsearch index to create.
            kwargs: create index key words arguments
        """

        vectorStore = BESVectorStore._bes_vector_store(embedding=embedding, **kwargs)

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
    def _bes_vector_store(
        embedding: Optional[Embeddings] = None, **kwargs: Any
    ) -> "BESVectorStore":
        index_name = kwargs.get("index_name")

        if index_name is None:
            raise ValueError("Please provide an index_name.")

        bes_url = kwargs.get("bes_url")
        if bes_url is None:
            raise ValueError("Please provided a valid bes connection url")

        return BESVectorStore(embedding=embedding, **kwargs)
