"""Wrapper around Elasticsearch vector database."""

from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.vectorstores.utils import DistanceStrategy
from abc import ABC, abstractmethod

import logging

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Callable,
    Union,
)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger()


class BaseRetrievalStrategy(ABC):
    @abstractmethod
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
        pass

    @abstractmethod
    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        pass

    def beforeIndexSetup(
        self, client: "Elasticsearch", text_field: str, vector_query_field: str
    ) -> None:
        pass

    def requireInference(self) -> bool:
        return True


class ApproxRetrievalStrategy(BaseRetrievalStrategy):
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

        # Case 2: Used when model has been deployed to Elasticsearch and can infer the query vector from the query text
        elif query and self.query_model_id:
            knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": self.query_model_id,  # use 'model_id' argument
                    "model_text": query,  # use 'query' argument
                }
            }

        else:
            raise ValueError(
                "You must provide an embedding function or a query_model_id to perform a similarity search."
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
        dims_length: int,
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
        logger.error(f"similarity {similarity}")
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

        if filter is None:
            queryBool = {"match_all": {}}
        else:
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
        dims_length: int,
        vector_query_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """Create the mapping for the Elasticsearch index."""

        return {
            "mappings": {
                "properties": {
                    vector_query_field: {"type": "dense_vector", "dims": dims_length},
                }
            }
        }


class SparseRetrievalStrategy(BaseRetrievalStrategy):
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

    def _getPipelineName(self) -> str:
        return f"{self.model_id}_sparse_embedding"

    def beforeIndexSetup(
        self, client: "Elasticsearch", vector_query_field: str, text_field: str
    ) -> None:
        # If model_id is provided, create a pipeline for the model
        if self.model_id:
            client.ingest.put_pipeline(
                id=self._getPipelineName(),
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
        vector_query_field: str,
        similarity: Union[DistanceStrategy, None],
        dims_length: Optional[int] = None,
    ) -> Dict:
        return {
            "mappings": {
                "properties": {
                    vector_query_field: {
                        "properties": {"tokens": {"type": "rank_features"}}
                    }
                }
            },
            "settings": {"default_pipeline": self._getPipelineName()},
        }

    def requireInference(self) -> bool:
        return False


class ElasticsearchStore(VectorStore):
    def __init__(
        self,
        index_name: str,
        embedding: Optional[Embeddings] = None,
        es_connection: Optional["Elasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        vector_query_field: str = "vector",
        query_field: str = "text",
        distance_strategy: Optional[DistanceStrategy] = DistanceStrategy.COSINE,
        strategy: BaseRetrievalStrategy = ApproxRetrievalStrategy(),
    ):
        self.embedding = embedding
        self.index_name = index_name
        self.query_field = query_field
        self.vector_query_field = vector_query_field
        self.distance_strategy = distance_strategy
        self.strategy = strategy

        if es_connection is not None:
            self.client = es_connection
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
    def connect_to_elasticsearch(
        es_url=None, cloud_id=None, api_key=None, username=None, password=None
    ):
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

        connection_params = {}

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

        es_client = elasticsearch.Elasticsearch(**connection_params)
        try:
            es_client.info()
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise e

        return es_client

    def similarity_search(
        self, query: str, filter: Optional[List[dict]] = None, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Pass through to `_search`, only returning the documents and not the scores.
        """

        results = self._search(query=query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Pass through to `_search including score`"""
        return self._search(query=query, k=k, filter=filter, **kwargs)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self._search(query_vector=embedding, k=k, filter=filter, **kwargs)

    def _search(
        self,
        query: Optional[str] = None,
        k: int = 4,
        query_vector: List[float] | None = None,
        fetch_k: int = 50,
        fields: Optional[List[str]] = [],
        filter: Optional[List[dict]] = None,
        custom_query: Optional[Callable[[Dict, str | None], Dict]] = None,
    ) -> List[Tuple[Document, float]]:
        if fields is None:
            fields = []

        if self.query_field not in fields:
            fields.append(self.query_field)

        fields.append("metadata")

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
            source=fields,  # type: ignore
        )

        hits = [hit for hit in response["hits"]["hits"]]  # type: ignore
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

    def _create_index_if_not_exists(
        self, index_name: str, dims_length: Optional[int] = None
    ) -> None:
        if self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists. Skipping creation.")

        else:
            if dims_length is None and self.strategy.requireInference():
                raise ValueError(
                    "Cannot create index without specifying dims_length when the index doesn't already exist. We infer dims_length from the first embedding. Check that you have provided an embedding function."
                )

            self.strategy.beforeIndexSetup(
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
        **kwargs: Any,
    ) -> List[str]:
        if self.embedding is not None:
            # If no search_type requires inference, we use the provided
            # embedding function to embed the texts.
            embeddings = self.embedding.embed_documents(list(texts))
            dims_length = len(embeddings[0])

            self._create_index_if_not_exists(
                index_name=self.index_name, dims_length=dims_length
            )

        else:
            # the search_type doesn't require inference, so we don't need to
            # embed the texts.
            embeddings = [None for _ in range(len(list(texts)))]

            self._create_index_if_not_exists(index_name=self.index_name)

        body = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}

            body.extend(
                [
                    {"index": {"_index": self.index_name}},
                    {"text": text, "vector": vector, "metadata": metadata},
                ]
            )

        if len(body) > 0:
            responses = self.client.bulk(operations=body)

            ids = [
                item["index"]["_id"]
                for item in responses["items"]
                if item["index"]["result"] == "created"
            ]

            return ids

        else:
            logger.debug("No texts to add to index")
            return []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "ElasticsearchStore":
        index_name = kwargs.get("index_name")

        if index_name is None:
            raise ValueError("Please provide an index_name.")

        es_connection = kwargs.get("es_connection")
        es_cloud_id = kwargs.get("es_cloud_id")
        es_url = kwargs.get("es_url")
        es_user = kwargs.get("es_user")
        es_password = kwargs.get("es_password")
        vector_query_field = kwargs.get("vector_query_field")
        query_field = kwargs.get("query_field")
        strategy = kwargs.get("strategy", ElasticsearchStore.ApproxRetrievalStrategy())

        optional_args = {}

        if vector_query_field is not None:
            optional_args["vector_query_field"] = vector_query_field

        if query_field is not None:
            optional_args["query_field"] = query_field

        elasticsearchStore = cls(
            index_name=index_name,
            embedding=embedding,
            es_url=es_url,
            es_connection=es_connection,
            es_cloud_id=es_cloud_id,
            es_user=es_user,
            es_password=es_password,
            strategy=strategy,
            **optional_args,
        )
        # Encode the provided texts and add them to the newly created index.
        elasticsearchStore.add_texts(
            texts,
            metadatas=metadatas,
            **optional_args,
        )

        return elasticsearchStore

    @staticmethod
    def ExactRetrievalStrategy() -> "ExactRetrievalStrategy":
        return ExactRetrievalStrategy()

    @staticmethod
    def ApproxRetrievalStrategy(
        query_model_id: Optional[str] = None,
        hybrid: Optional[bool] = False,
    ) -> "ApproxRetrievalStrategy":
        return ApproxRetrievalStrategy(query_model_id=query_model_id, hybrid=hybrid)

    @staticmethod
    def SparseVectorRetrievalStrategy(
        model_id: Optional[str] = None,
    ) -> "SparseRetrievalStrategy":
        return SparseRetrievalStrategy(model_id=model_id)
