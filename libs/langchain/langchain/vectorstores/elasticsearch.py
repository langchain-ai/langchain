import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.utils import DistanceStrategy

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class BaseRetrievalStrategy(ABC):
    """Base class for `Elasticsearch` retrieval strategies."""

    @abstractmethod
    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        *,
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: List[dict],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """
        Executes when a search is performed on the store.

        Args:
            query_vector: The query vector,
                          or None if not using vector-based query.
            query: The text query, or None if not using text-based query.
            k: The total number of results to retrieve.
            fetch_k: The number of results to fetch initially.
            vector_query_field: The field containing the vector
                                representations in the index.
            text_field: The field containing the text data in the index.
            filter: List of filter clauses to apply to the query.
            similarity: The similarity strategy to use, or None if not using one.

        Returns:
            Dict: The Elasticsearch query body.
        """

    @abstractmethod
    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """
        Executes when the index is created.

        Args:
            dims_length: Numeric length of the embedding vectors,
                        or None if not using vector-based query.
            vector_query_field: The field containing the vector
                                representations in the index.
            similarity: The similarity strategy to use,
                        or None if not using one.

        Returns:
            Dict: The Elasticsearch settings and mappings for the strategy.
        """

    def before_index_setup(
        self, client: "Elasticsearch", text_field: str, vector_query_field: str
    ) -> None:
        """
        Executes before the index is created. Used for setting up
        any required Elasticsearch resources like a pipeline.

        Args:
            client: The Elasticsearch client.
            text_field: The field containing the text data in the index.
            vector_query_field: The field containing the vector
                                representations in the index.
        """

    def require_inference(self) -> bool:
        """
        Returns whether or not the strategy requires inference
        to be performed on the text before it is added to the index.

        Returns:
            bool: Whether or not the strategy requires inference
            to be performed on the text before it is added to the index.
        """
        return True


class ApproxRetrievalStrategy(BaseRetrievalStrategy):
    """Approximate retrieval strategy using the `HNSW` algorithm."""

    def __init__(
        self,
        query_model_id: Optional[str] = None,
        hybrid: Optional[bool] = False,
    ):
        self.query_model_id = query_model_id
        self.hybrid = hybrid

    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: List[dict],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        knn = {
            "filter": filter,
            "field": vector_query_field,
            "k": k,
            "num_candidates": fetch_k,
        }

        # Embedding provided via the embedding function
        if query_vector and not self.query_model_id:
            knn["query_vector"] = query_vector

        # Case 2: Used when model has been deployed to
        # Elasticsearch and can infer the query vector from the query text
        elif query and self.query_model_id:
            knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": self.query_model_id,  # use 'model_id' argument
                    "model_text": query,  # use 'query' argument
                }
            }

        else:
            raise ValueError(
                "You must provide an embedding function or a"
                " query_model_id to perform a similarity search."
            )

        # If hybrid, add a query to the knn query
        # RRF is used to even the score from the knn query and text query
        if self.hybrid:
            return {
                "knn": knn,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    text_field: {
                                        "query": query,
                                    }
                                }
                            }
                        ],
                        "filter": filter,
                    }
                },
                "rank": {"rrf": {}},
            }
        else:
            return {"knn": knn}

    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """Create the mapping for the Elasticsearch index."""

        if similarity is DistanceStrategy.COSINE:
            similarityAlgo = "cosine"
        elif similarity is DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarityAlgo = "l2_norm"
        elif similarity is DistanceStrategy.DOT_PRODUCT:
            similarityAlgo = "dot_product"
        else:
            raise ValueError(f"Similarity {similarity} not supported.")

        return {
            "mappings": {
                "properties": {
                    vector_query_field: {
                        "type": "dense_vector",
                        "dims": dims_length,
                        "index": True,
                        "similarity": similarityAlgo,
                    },
                }
            }
        }


class ExactRetrievalStrategy(BaseRetrievalStrategy):
    """Exact retrieval strategy using the `script_score` query."""

    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: Union[List[dict], None],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        if similarity is DistanceStrategy.COSINE:
            similarityAlgo = (
                f"cosineSimilarity(params.query_vector, '{vector_query_field}') + 1.0"
            )
        elif similarity is DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarityAlgo = (
                f"1 / (1 + l2norm(params.query_vector, '{vector_query_field}'))"
            )
        elif similarity is DistanceStrategy.DOT_PRODUCT:
            similarityAlgo = f"""
            double value = dotProduct(params.query_vector, '{vector_query_field}');
            return sigmoid(1, Math.E, -value); 
            """
        else:
            raise ValueError(f"Similarity {similarity} not supported.")

        queryBool: Dict = {"match_all": {}}
        if filter:
            queryBool = {"bool": {"filter": filter}}

        return {
            "query": {
                "script_score": {
                    "query": queryBool,
                    "script": {
                        "source": similarityAlgo,
                        "params": {"query_vector": query_vector},
                    },
                },
            }
        }

    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """Create the mapping for the Elasticsearch index."""

        return {
            "mappings": {
                "properties": {
                    vector_query_field: {
                        "type": "dense_vector",
                        "dims": dims_length,
                        "index": False,
                    },
                }
            }
        }


class SparseRetrievalStrategy(BaseRetrievalStrategy):
    """Sparse retrieval strategy using the `text_expansion` processor."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or ".elser_model_1"

    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: List[dict],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        return {
            "query": {
                "bool": {
                    "must": [
                        {
                            "text_expansion": {
                                f"{vector_query_field}.tokens": {
                                    "model_id": self.model_id,
                                    "model_text": query,
                                }
                            }
                        }
                    ],
                    "filter": filter,
                }
            }
        }

    def _get_pipeline_name(self) -> str:
        return f"{self.model_id}_sparse_embedding"

    def before_index_setup(
        self, client: "Elasticsearch", text_field: str, vector_query_field: str
    ) -> None:
        # If model_id is provided, create a pipeline for the model
        if self.model_id:
            client.ingest.put_pipeline(
                id=self._get_pipeline_name(),
                description="Embedding pipeline for langchain vectorstore",
                processors=[
                    {
                        "inference": {
                            "model_id": self.model_id,
                            "target_field": vector_query_field,
                            "field_map": {text_field: "text_field"},
                            "inference_config": {
                                "text_expansion": {"results_field": "tokens"}
                            },
                        }
                    }
                ],
            )

    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        return {
            "mappings": {
                "properties": {
                    vector_query_field: {
                        "properties": {"tokens": {"type": "rank_features"}}
                    }
                }
            },
            "settings": {"default_pipeline": self._get_pipeline_name()},
        }

    def require_inference(self) -> bool:
        return False


class ElasticsearchStore(VectorStore):
    """`Elasticsearch` vector store.

    Example:
        .. code-block:: python

            from langchain.vectorstores import ElasticsearchStore
            from langchain.embeddings.openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vectorstore = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_url="http://localhost:9200"
            )

    Args:
        index_name: Name of the Elasticsearch index to create.
        es_url: URL of the Elasticsearch instance to connect to.
        cloud_id: Cloud ID of the Elasticsearch instance to connect to.
        es_user: Username to use when connecting to Elasticsearch.
        es_password: Password to use when connecting to Elasticsearch.
        es_api_key: API key to use when connecting to Elasticsearch.
        es_connection: Optional pre-existing Elasticsearch connection.
        vector_query_field: Optional. Name of the field to store
                            the embedding vectors in.
        query_field: Optional. Name of the field to store the texts in.
        strategy: Optional. Retrieval strategy to use when searching the index.
                 Defaults to ApproxRetrievalStrategy. Can be one of
                 ExactRetrievalStrategy, ApproxRetrievalStrategy,
                 or SparseRetrievalStrategy.
        distance_strategy: Optional. Distance strategy to use when
                            searching the index.
                            Defaults to COSINE. Can be one of COSINE,
                            EUCLIDEAN_DISTANCE, or DOT_PRODUCT.

    If you want to use a cloud hosted Elasticsearch instance, you can pass in the
    cloud_id argument instead of the es_url argument.

    Example:
        .. code-block:: python

            from langchain.vectorstores import ElasticsearchStore
            from langchain.embeddings.openai import OpenAIEmbeddings

            vectorstore = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_cloud_id="<cloud_id>"
                es_user="elastic",
                es_password="<password>"
            )

    You can also connect to an existing Elasticsearch instance by passing in a
    pre-existing Elasticsearch connection via the es_connection argument.

    Example:
        .. code-block:: python

            from langchain.vectorstores import ElasticsearchStore
            from langchain.embeddings.openai import OpenAIEmbeddings

            from elasticsearch import Elasticsearch

            es_connection = Elasticsearch("http://localhost:9200")

            vectorstore = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_connection=es_connection
            )

    ElasticsearchStore by default uses the ApproxRetrievalStrategy, which uses the
    HNSW algorithm to perform approximate nearest neighbor search. This is the
    fastest and most memory efficient algorithm.

    If you want to use the Brute force / Exact strategy for searching vectors, you
    can pass in the ExactRetrievalStrategy to the ElasticsearchStore constructor.

    Example:
        .. code-block:: python

            from langchain.vectorstores import ElasticsearchStore
            from langchain.embeddings.openai import OpenAIEmbeddings

            vectorstore = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_url="http://localhost:9200",
                strategy=ElasticsearchStore.ExactRetrievalStrategy()
            )

    Both strategies require that you know the similarity metric you want to use
    when creating the index. The default is cosine similarity, but you can also
    use dot product or euclidean distance.

    Example:
        .. code-block:: python

            from langchain.vectorstores import ElasticsearchStore
            from langchain.embeddings.openai import OpenAIEmbeddings
            from langchain.vectorstores.utils import DistanceStrategy

            vectorstore = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_url="http://localhost:9200",
                distance_strategy="DOT_PRODUCT"
            )

    """

    def __init__(
        self,
        index_name: str,
        *,
        embedding: Optional[Embeddings] = None,
        es_connection: Optional["Elasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        vector_query_field: str = "vector",
        query_field: str = "text",
        distance_strategy: Optional[
            Literal[
                DistanceStrategy.COSINE,
                DistanceStrategy.DOT_PRODUCT,
                DistanceStrategy.EUCLIDEAN_DISTANCE,
            ]
        ] = None,
        strategy: BaseRetrievalStrategy = ApproxRetrievalStrategy(),
    ):
        self.embedding = embedding
        self.index_name = index_name
        self.query_field = query_field
        self.vector_query_field = vector_query_field
        self.distance_strategy = (
            DistanceStrategy.COSINE
            if distance_strategy is None
            else DistanceStrategy[distance_strategy]
        )
        self.strategy = strategy

        if es_connection is not None:
            self.client = es_connection.options(
                headers={"user-agent": self.get_user_agent()}
            )
        elif es_url is not None or es_cloud_id is not None:
            self.client = ElasticsearchStore.connect_to_elasticsearch(
                es_url=es_url,
                username=es_user,
                password=es_password,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
            )
        else:
            raise ValueError(
                """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
            )

    @staticmethod
    def get_user_agent() -> str:
        from langchain import __version__

        return f"langchain-py-vs/{__version__}"

    @staticmethod
    def connect_to_elasticsearch(
        *,
        es_url: Optional[str] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
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

        if es_url and cloud_id:
            raise ValueError(
                "Both es_url and cloud_id are defined. Please provide only one."
            )

        connection_params: Dict[str, Any] = {}

        if es_url:
            connection_params["hosts"] = [es_url]
        elif cloud_id:
            connection_params["cloud_id"] = cloud_id
        else:
            raise ValueError("Please provide either elasticsearch_url or cloud_id.")

        if api_key:
            connection_params["api_key"] = api_key
        elif username and password:
            connection_params["basic_auth"] = (username, password)

        es_client = elasticsearch.Elasticsearch(
            **connection_params,
            headers={"user-agent": ElasticsearchStore.get_user_agent()},
        )
        try:
            es_client.info()
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise e

        return es_client

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return Elasticsearch documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the query,
            in descending order of similarity.
        """

        results = self._search(query=query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return Elasticsearch documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the query and score for each
        """
        return self._search(query=query, k=k, filter=filter, **kwargs)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Elasticsearch documents most similar to query, along with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the embedding and score for each
        """
        return self._search(query_vector=embedding, k=k, filter=filter, **kwargs)

    def _search(
        self,
        query: Optional[str] = None,
        k: int = 4,
        query_vector: Union[List[float], None] = None,
        fetch_k: int = 50,
        fields: Optional[List[str]] = None,
        filter: Optional[List[dict]] = None,
        custom_query: Optional[Callable[[Dict, Union[str, None]], Dict]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return Elasticsearch documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            query_vector: Embedding to look up documents similar to.
            fetch_k: Number of candidates to fetch from each shard.
                    Defaults to 50.
            fields: List of fields to return from Elasticsearch.
                    Defaults to only returning the text field.
            filter: Array of Elasticsearch filter clauses to apply to the query.
            custom_query: Function to modify the Elasticsearch
                         query body before it is sent to Elasticsearch.

        Returns:
            List of Documents most similar to the query and score for each
        """
        if fields is None:
            fields = ["metadata"]

        if self.query_field not in fields:
            fields.append(self.query_field)

        if self.embedding and query is not None:
            query_vector = self.embedding.embed_query(query)

        query_body = self.strategy.query(
            query_vector=query_vector,
            query=query,
            k=k,
            fetch_k=fetch_k,
            vector_query_field=self.vector_query_field,
            text_field=self.query_field,
            filter=filter or [],
            similarity=self.distance_strategy,
        )

        logger.debug(f"Query body: {query_body}")

        if custom_query is not None:
            query_body = custom_query(query_body, query)
            logger.debug(f"Calling custom_query, Query body now: {query_body}")

        # Perform the kNN search on the Elasticsearch index and return the results.
        response = self.client.search(
            index=self.index_name,
            **query_body,
            size=k,
            source=fields,
        )

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

    def delete(
        self,
        ids: Optional[List[str]] = None,
        refresh_indices: Optional[bool] = True,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents from the Elasticsearch index.

        Args:
            ids: List of ids of documents to delete.
            refresh_indices: Whether to refresh the index
                            after deleting documents. Defaults to True.
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
                bulk(self.client, body, refresh=refresh_indices, ignore_status=404)
                logger.debug(f"Deleted {len(body)} texts from index")

                return True
            except BulkIndexError as e:
                logger.error(f"Error deleting texts: {e}")
                firstError = e.errors[0].get("index", {}).get("error", {})
                logger.error(f"First error reason: {firstError.get('reason')}")
                raise e

        else:
            logger.debug("No texts to delete from index")
            return False

    def _create_index_if_not_exists(
        self, index_name: str, dims_length: Optional[int] = None
    ) -> None:
        """Create the Elasticsearch index if it doesn't already exist.

        Args:
            index_name: Name of the Elasticsearch index to create.
            dims_length: Length of the embedding vectors.
        """

        if self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists. Skipping creation.")

        else:
            if dims_length is None and self.strategy.require_inference():
                raise ValueError(
                    "Cannot create index without specifying dims_length "
                    "when the index doesn't already exist. We infer "
                    "dims_length from the first embedding. Check that "
                    "you have provided an embedding function."
                )

            self.strategy.before_index_setup(
                client=self.client,
                text_field=self.query_field,
                vector_query_field=self.vector_query_field,
            )

            indexSettings = self.strategy.index(
                vector_query_field=self.vector_query_field,
                dims_length=dims_length,
                similarity=self.distance_strategy,
            )
            logger.debug(
                f"Creating index {index_name} with mappings {indexSettings['mappings']}"
            )
            self.client.indices.create(index=index_name, **indexSettings)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        refresh_indices: bool = True,
        create_index_if_not_exists: bool = True,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            refresh_indices: Whether to refresh the Elasticsearch indices
                            after adding the texts.
            create_index_if_not_exists: Whether to create the Elasticsearch
                                        index if it doesn't already exist.
            *bulk_kwargs: Additional arguments to pass to Elasticsearch bulk.
                - chunk_size: Optional. Number of texts to add to the
                    index at a time. Defaults to 500.

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
        bulk_kwargs = bulk_kwargs or {}
        embeddings = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        requests = []

        if self.embedding is not None:
            # If no search_type requires inference, we use the provided
            # embedding function to embed the texts.
            embeddings = self.embedding.embed_documents(list(texts))
            dims_length = len(embeddings[0])

            if create_index_if_not_exists:
                self._create_index_if_not_exists(
                    index_name=self.index_name, dims_length=dims_length
                )

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
            # the search_type doesn't require inference, so we don't need to
            # embed the texts.
            if create_index_if_not_exists:
                self._create_index_if_not_exists(index_name=self.index_name)

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
                    self.client,
                    requests,
                    stats_only=True,
                    refresh=refresh_indices,
                    **bulk_kwargs,
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

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> "ElasticsearchStore":
        """Construct ElasticsearchStore wrapper from raw documents.

        Example:
            .. code-block:: python

                from langchain.vectorstores import ElasticsearchStore
                from langchain.embeddings.openai import OpenAIEmbeddings

                db = ElasticsearchStore.from_texts(
                    texts,
                    // embeddings optional if using
                    // a strategy that doesn't require inference
                    embeddings,
                    index_name="langchain-demo",
                    es_url="http://localhost:9200"
                )

        Args:
            texts: List of texts to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
            metadatas: Optional list of metadatas associated with the texts.
            index_name: Name of the Elasticsearch index to create.
            es_url: URL of the Elasticsearch instance to connect to.
            cloud_id: Cloud ID of the Elasticsearch instance to connect to.
            es_user: Username to use when connecting to Elasticsearch.
            es_password: Password to use when connecting to Elasticsearch.
            es_api_key: API key to use when connecting to Elasticsearch.
            es_connection: Optional pre-existing Elasticsearch connection.
            vector_query_field: Optional. Name of the field to
                                store the embedding vectors in.
            query_field: Optional. Name of the field to store the texts in.
            distance_strategy: Optional. Name of the distance
                                strategy to use. Defaults to "COSINE".
                                can be one of "COSINE",
                                "EUCLIDEAN_DISTANCE", "DOT_PRODUCT".
            bulk_kwargs: Optional. Additional arguments to pass to
                        Elasticsearch bulk.
        """

        elasticsearchStore = ElasticsearchStore._create_cls_from_kwargs(
            embedding=embedding, **kwargs
        )

        # Encode the provided texts and add them to the newly created index.
        elasticsearchStore.add_texts(
            texts, metadatas=metadatas, bulk_kwargs=bulk_kwargs
        )

        return elasticsearchStore

    @staticmethod
    def _create_cls_from_kwargs(
        embedding: Optional[Embeddings] = None, **kwargs: Any
    ) -> "ElasticsearchStore":
        index_name = kwargs.get("index_name")

        if index_name is None:
            raise ValueError("Please provide an index_name.")

        es_connection = kwargs.get("es_connection")
        es_cloud_id = kwargs.get("es_cloud_id")
        es_url = kwargs.get("es_url")
        es_user = kwargs.get("es_user")
        es_password = kwargs.get("es_password")
        es_api_key = kwargs.get("es_api_key")
        vector_query_field = kwargs.get("vector_query_field")
        query_field = kwargs.get("query_field")
        distance_strategy = kwargs.get("distance_strategy")
        strategy = kwargs.get("strategy", ElasticsearchStore.ApproxRetrievalStrategy())

        optional_args = {}

        if vector_query_field is not None:
            optional_args["vector_query_field"] = vector_query_field

        if query_field is not None:
            optional_args["query_field"] = query_field

        return ElasticsearchStore(
            index_name=index_name,
            embedding=embedding,
            es_url=es_url,
            es_connection=es_connection,
            es_cloud_id=es_cloud_id,
            es_user=es_user,
            es_password=es_password,
            es_api_key=es_api_key,
            strategy=strategy,
            distance_strategy=distance_strategy,
            **optional_args,
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> "ElasticsearchStore":
        """Construct ElasticsearchStore wrapper from documents.

        Example:
            .. code-block:: python

                from langchain.vectorstores import ElasticsearchStore
                from langchain.embeddings.openai import OpenAIEmbeddings

                db = ElasticsearchStore.from_documents(
                    texts,
                    embeddings,
                    index_name="langchain-demo",
                    es_url="http://localhost:9200"
                )

        Args:
            texts: List of texts to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
                      Do not provide if using a strategy
                      that doesn't require inference.
            metadatas: Optional list of metadatas associated with the texts.
            index_name: Name of the Elasticsearch index to create.
            es_url: URL of the Elasticsearch instance to connect to.
            cloud_id: Cloud ID of the Elasticsearch instance to connect to.
            es_user: Username to use when connecting to Elasticsearch.
            es_password: Password to use when connecting to Elasticsearch.
            es_api_key: API key to use when connecting to Elasticsearch.
            es_connection: Optional pre-existing Elasticsearch connection.
            vector_query_field: Optional. Name of the field
                                to store the embedding vectors in.
            query_field: Optional. Name of the field to store the texts in.
            bulk_kwargs: Optional. Additional arguments to pass to
                        Elasticsearch bulk.
        """

        elasticsearchStore = ElasticsearchStore._create_cls_from_kwargs(
            embedding=embedding, **kwargs
        )
        # Encode the provided texts and add them to the newly created index.
        elasticsearchStore.add_documents(documents, bulk_kwargs=bulk_kwargs)

        return elasticsearchStore

    @staticmethod
    def ExactRetrievalStrategy() -> "ExactRetrievalStrategy":
        """Used to perform brute force / exact
        nearest neighbor search via script_score."""
        return ExactRetrievalStrategy()

    @staticmethod
    def ApproxRetrievalStrategy(
        query_model_id: Optional[str] = None,
        hybrid: Optional[bool] = False,
    ) -> "ApproxRetrievalStrategy":
        """Used to perform approximate nearest neighbor search
        using the HNSW algorithm.

        At build index time, this strategy will create a
        dense vector field in the index and store the
        embedding vectors in the index.

        At query time, the text will either be embedded using the
        provided embedding function or the query_model_id
        will be used to embed the text using the model
        deployed to Elasticsearch.

        if query_model_id is used, do not provide an embedding function.

        Args:
            query_model_id: Optional. ID of the model to use to
                            embed the query text within the stack. Requires
                            embedding model to be deployed to Elasticsearch.
            hybrid: Optional. If True, will perform a hybrid search
                    using both the knn query and a text query.
                    Defaults to False.
        """
        return ApproxRetrievalStrategy(query_model_id=query_model_id, hybrid=hybrid)

    @staticmethod
    def SparseVectorRetrievalStrategy(
        model_id: Optional[str] = None,
    ) -> "SparseRetrievalStrategy":
        """Used to perform sparse vector search via text_expansion.
        Used for when you want to use ELSER model to perform document search.

        At build index time, this strategy will create a pipeline that
        will embed the text using the ELSER model and store the
        resulting tokens in the index.

        At query time, the text will be embedded using the ELSER
        model and the resulting tokens will be used to
        perform a text_expansion query.

        Args:
            model_id: Optional. Default is ".elser_model_1".
                    ID of the model to use to embed the query text
                    within the stack. Requires embedding model to be
                    deployed to Elasticsearch.
        """
        return SparseRetrievalStrategy(model_id=model_id)
