"""Wrapper around Elasticsearch vector database."""

from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.vectorstores.utils import DistanceStrategy
from abc import ABC, abstractmethod

import logging

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Callable

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger()


class BaseRetrievalStrategy(ABC):
    @abstractmethod
    def query(
        self,
        query_vector: List[float] | None,
        query: str | None,
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: dict,
        similarity: DistanceStrategy | None,
    ) -> Dict:
        pass

    @abstractmethod
    def mapping(
        self,
        dims_length: int | None,
        vector_query_field: str,
        similarity: DistanceStrategy | None,
    ) -> Dict:
        pass

    def shouldInfer(self) -> bool:
        return True


class ApproxRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(
        self,
        model_id: Optional[str] = None,
        dim_length: Optional[int] = None,
    ):
        self.model_id = model_id
        self.dim_length = dim_length

    def query(
        self,
        query_vector: List[float] | None,
        query: str | None,
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: dict,
        similarity: DistanceStrategy | None,
    ) -> Dict:
        knn: Dict = {
            # "filter": filter,
            "field": vector_query_field,
            "k": k,
            "num_candidates": fetch_k,
        }

        # Case 1: `query_vector` is provided, but not `model_id` -> use query_vector
        if query_vector and not self.model_id:
            knn["query_vector"] = query_vector

        # Case 2: `query` and `model_id` are provided, -> use query_vector_builder
        elif query and self.model_id:
            knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": self.model_id,  # use 'model_id' argument
                    "model_text": query,  # use 'query' argument
                }
            }

        else:
            raise ValueError(
                "Either `query_vector` or `model_id` must be provided, but not both."
            )

        return {"knn": knn}

    def mapping(
        self,
        dims_length: int,
        vector_query_field: str,
        similarity: DistanceStrategy | None,
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
            "properties": {
                vector_query_field: {
                    "type": "dense_vector",
                    "dims": self.dim_length or dims_length,
                    "index": True,
                    "similarity": similarityAlgo,
                },
            }
        }

    def shouldInfer(self) -> bool:
        return self.model_id is None

    @staticmethod
    def use() -> "ApproxRetrievalStrategy":
        return ApproxRetrievalStrategy()

    @staticmethod
    def useModel(model_id: str, dims_length: int) -> "ApproxRetrievalStrategy":
        return ApproxRetrievalStrategy(model_id=model_id, dim_length=dims_length)


class CustomRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, query: Callable, mapping: Callable, shouldInfer: bool):
        self.query = query
        self.mapping = mapping
        self.si = shouldInfer

    def query(
        self,
        **args: Any,
    ) -> Dict:
        return self.query(**args)

    def mapping(
        self,
        **args: Any,
    ) -> Dict:
        return self.mapping(**args)

    def shouldInfer(self) -> bool:
        return self.si

    @staticmethod
    def use(
        query: Callable, mapping: Callable, shouldInfer: bool
    ) -> "CustomRetrievalStrategy":
        return CustomRetrievalStrategy(
            query=query, mapping=mapping, shouldInfer=shouldInfer
        )


class ExactRetrievalStrategy(BaseRetrievalStrategy):
    def query(
        self,
        query_vector: List[float] | None,
        query: str | None,
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: dict | None,
        similarity: DistanceStrategy | None,
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

        return {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": similarityAlgo,
                        "params": {"query_vector": query_vector},
                    },
                },
            }
        }

    def mapping(
        self,
        dims_length: int,
        vector_query_field: str,
        similarity: DistanceStrategy | None,
    ) -> Dict:
        """Create the mapping for the Elasticsearch index."""

        return {
            "properties": {
                vector_query_field: {"type": "dense_vector", "dims": dims_length},
            }
        }

    @staticmethod
    def use() -> "ExactRetrievalStrategy":
        return ExactRetrievalStrategy()


class ElasticsearchStore(VectorStore):
    STRATEGIES = {
        "approx": ApproxRetrievalStrategy,
        "exact": ExactRetrievalStrategy,
        "custom": CustomRetrievalStrategy,
    }

    def __init__(
        self,
        index_name: str,
        embedding: Embeddings,
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
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Pass through to `_search`, only returning the documents and not the scores.
        """
        results = self._search(query=query, k=k, **kwargs)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Pass through to `_search including score`"""
        return self._search(query=query, k=k, **kwargs)

    def _search(
        self,
        query: str | None = None,
        k: int = 10,
        query_vector: List[float] | None = None,
        fetch_k: int = 100,
        fields: List[str] = [],
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        if self.query_field not in fields:
            fields.append(self.query_field)

        fields.append("metadata")

        if self.strategy.shouldInfer() and query is not None:
            query_vector = self.embedding.embed_query(query)

        query_body = self.strategy.query(
            query_vector=query_vector,
            query=query,
            k=k,
            fetch_k=fetch_k,
            vector_query_field=self.vector_query_field,
            text_field=self.query_field,
            filter=filter or {},
            similarity=self.distance_strategy,
        )

        logger.debug(f"Query body: {query_body}")

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
            logger.debug(f"Index {index_name} already exists.")

        else:
            mapping = self.strategy.mapping(
                vector_query_field=self.vector_query_field,
                dims_length=dims_length,
                similarity=self.distance_strategy,
            )
            logger.debug(f"Creating index {index_name} with mapping {mapping}")
            self.client.indices.create(index=index_name, mappings=mapping)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if self.strategy.shouldInfer():
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
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
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
        strategy = kwargs.get("strategy", ApproxRetrievalStrategy.use())

        optional_args = {}

        if vector_query_field is not None:
            optional_args["vector_query_field"] = vector_query_field

        if query_field is not None:
            optional_args["query_field"] = query_field

        elasticsearch = cls(
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
        elasticsearch.add_texts(
            texts,
            metadatas=metadatas,
            **optional_args,
        )

        return elasticsearch
