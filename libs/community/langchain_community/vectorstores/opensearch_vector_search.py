from __future__ import annotations

import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)
IMPORT_ASYNC_OPENSEARCH_PY_ERROR = """
Could not import AsyncOpenSearch.
Please install it with `pip install opensearch-py`."""

SCRIPT_SCORING_SEARCH = "script_scoring"
PAINLESS_SCRIPTING_SEARCH = "painless_scripting"
MATCH_ALL_QUERY = {"match_all": {}}  # type: Dict
HYBRID_SEARCH = "hybrid_search"

if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch, OpenSearch


def _get_opensearch_client(opensearch_url: str, **kwargs: Any) -> OpenSearch:
    """Get OpenSearch client from the opensearch_url, otherwise raise error."""
    try:
        from opensearchpy import OpenSearch

        client = OpenSearch(opensearch_url, **kwargs)
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    except ValueError as e:
        raise ImportError(
            f"OpenSearch client string provided is not in proper format. "
            f"Got error: {e} "
        )
    return client


def _get_async_opensearch_client(opensearch_url: str, **kwargs: Any) -> AsyncOpenSearch:
    """Get AsyncOpenSearch client from the opensearch_url, otherwise raise error."""
    try:
        from opensearchpy import AsyncOpenSearch

        client = AsyncOpenSearch(opensearch_url, **kwargs)
    except ImportError:
        raise ImportError(IMPORT_ASYNC_OPENSEARCH_PY_ERROR)
    except ValueError as e:
        raise ImportError(
            f"AsyncOpenSearch client string provided is not in proper format. "
            f"Got error: {e} "
        )
    return client


def _validate_embeddings_and_bulk_size(embeddings_length: int, bulk_size: int) -> None:
    """Validate Embeddings Length and Bulk Size."""
    if embeddings_length == 0:
        raise RuntimeError("Embeddings size is zero")
    if bulk_size < embeddings_length:
        raise RuntimeError(
            f"The embeddings count, {embeddings_length} is more than the "
            f"[bulk_size], {bulk_size}. Increase the value of [bulk_size]."
        )


def _validate_aoss_with_engines(is_aoss: bool, engine: str) -> None:
    """Validate AOSS with the engine."""
    if is_aoss and engine != "nmslib" and engine != "faiss":
        raise ValueError(
            "Amazon OpenSearch Service Serverless only "
            "supports `nmslib` or `faiss` engines"
        )


def _is_aoss_enabled(http_auth: Any) -> bool:
    """Check if the service is http_auth is set as `aoss`."""
    if (
        http_auth is not None
        and hasattr(http_auth, "service")
        and http_auth.service == "aoss"
    ):
        return True
    return False


def _bulk_ingest_embeddings(
    client: OpenSearch,
    index_name: str,
    embeddings: List[List[float]],
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    vector_field: str = "vector_field",
    text_field: str = "text",
    mapping: Optional[Dict] = None,
    max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
    is_aoss: bool = False,
) -> List[str]:
    """Bulk Ingest Embeddings into given index."""
    if not mapping:
        mapping = dict()
    try:
        from opensearchpy.exceptions import NotFoundError
        from opensearchpy.helpers import bulk
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)

    requests = []
    return_ids = []
    mapping = mapping

    try:
        client.indices.get(index=index_name)
    except NotFoundError:
        client.indices.create(index=index_name, body=mapping)

    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas else {}
        _id = ids[i] if ids else str(uuid.uuid4())
        request = {
            "_op_type": "index",
            "_index": index_name,
            vector_field: embeddings[i],
            text_field: text,
            "metadata": metadata,
        }
        if is_aoss:
            request["id"] = _id
        else:
            request["_id"] = _id
        requests.append(request)
        return_ids.append(_id)
    bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
    if not is_aoss:
        client.indices.refresh(index=index_name)
    return return_ids


async def _abulk_ingest_embeddings(
    client: AsyncOpenSearch,
    index_name: str,
    embeddings: List[List[float]],
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    vector_field: str = "vector_field",
    text_field: str = "text",
    mapping: Optional[Dict] = None,
    max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
    is_aoss: bool = False,
) -> List[str]:
    """Bulk Ingest Embeddings into given index asynchronously using AsyncOpenSearch."""
    if not mapping:
        mapping = dict()

    try:
        from opensearchpy.exceptions import NotFoundError
        from opensearchpy.helpers import async_bulk
    except ImportError:
        raise ImportError(IMPORT_ASYNC_OPENSEARCH_PY_ERROR)

    requests = []
    return_ids = []

    try:
        await client.indices.get(index=index_name)
    except NotFoundError:
        await client.indices.create(index=index_name, body=mapping)

    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas else {}
        _id = ids[i] if ids else str(uuid.uuid4())
        request = {
            "_op_type": "index",
            "_index": index_name,
            vector_field: embeddings[i],
            text_field: text,
            "metadata": metadata,
        }
        if is_aoss:
            request["id"] = _id
        else:
            request["_id"] = _id
        requests.append(request)
        return_ids.append(_id)

    await async_bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
    if not is_aoss:
        await client.indices.refresh(index=index_name)

    return return_ids


def _default_scripting_text_mapping(
    dim: int,
    vector_field: str = "vector_field",
) -> Dict[str, Any]:
    """For Painless Scripting or Script Scoring,the default mapping to create index."""
    return {
        "mappings": {
            "properties": {
                vector_field: {"type": "knn_vector", "dimension": dim},
            }
        }
    }


def _default_text_mapping(
    dim: int,
    engine: str = "nmslib",
    space_type: str = "l2",
    ef_search: int = 512,
    ef_construction: int = 512,
    m: int = 16,
    vector_field: str = "vector_field",
) -> Dict[str, Any]:
    """For Approximate k-NN Search, this is the default mapping to create index."""
    return {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": ef_search}},
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": space_type,
                        "engine": engine,
                        "parameters": {"ef_construction": ef_construction, "m": m},
                    },
                }
            }
        },
    }


def _default_approximate_search_query(
    query_vector: List[float],
    k: int = 4,
    vector_field: str = "vector_field",
    score_threshold: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """For Approximate k-NN Search, this is the default query."""
    return {
        "size": k,
        "min_score": score_threshold,
        "query": {"knn": {vector_field: {"vector": query_vector, "k": k}}},
    }


def _approximate_search_query_with_boolean_filter(
    query_vector: List[float],
    boolean_filter: Dict,
    k: int = 4,
    vector_field: str = "vector_field",
    subquery_clause: str = "must",
    score_threshold: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """For Approximate k-NN Search, with Boolean Filter."""
    return {
        "size": k,
        "min_score": score_threshold,
        "query": {
            "bool": {
                "filter": boolean_filter,
                subquery_clause: [
                    {"knn": {vector_field: {"vector": query_vector, "k": k}}}
                ],
            }
        },
    }


def _approximate_search_query_with_efficient_filter(
    query_vector: List[float],
    efficient_filter: Dict,
    k: int = 4,
    vector_field: str = "vector_field",
    score_threshold: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """For Approximate k-NN Search, with Efficient Filter for Lucene and
    Faiss Engines."""
    search_query = _default_approximate_search_query(
        query_vector, k=k, vector_field=vector_field, score_threshold=score_threshold
    )
    search_query["query"]["knn"][vector_field]["filter"] = efficient_filter
    return search_query


def _default_script_query(
    query_vector: List[float],
    k: int = 4,
    space_type: str = "l2",
    pre_filter: Optional[Dict] = None,
    vector_field: str = "vector_field",
    score_threshold: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """For Script Scoring Search, this is the default query."""

    if not pre_filter:
        pre_filter = MATCH_ALL_QUERY

    return {
        "size": k,
        "min_score": score_threshold,
        "query": {
            "script_score": {
                "query": pre_filter,
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": vector_field,
                        "query_value": query_vector,
                        "space_type": space_type,
                    },
                },
            }
        },
    }


def __get_painless_scripting_source(
    space_type: str, vector_field: str = "vector_field"
) -> str:
    """For Painless Scripting, it returns the script source based on space type."""
    source_value = (
        "(1.0 + " + space_type + "(params.query_value, doc['" + vector_field + "']))"
    )
    if space_type == "cosineSimilarity":
        return source_value
    else:
        return "1/" + source_value


def _default_painless_scripting_query(
    query_vector: List[float],
    k: int = 4,
    space_type: str = "l2Squared",
    pre_filter: Optional[Dict] = None,
    vector_field: str = "vector_field",
    score_threshold: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """For Painless Scripting Search, this is the default query."""

    if not pre_filter:
        pre_filter = MATCH_ALL_QUERY

    source = __get_painless_scripting_source(space_type, vector_field=vector_field)
    return {
        "size": k,
        "min_score": score_threshold,
        "query": {
            "script_score": {
                "query": pre_filter,
                "script": {
                    "source": source,
                    "params": {
                        "field": vector_field,
                        "query_value": query_vector,
                    },
                },
            }
        },
    }


def _default_hybrid_search_query(
    query_text: str, query_vector: List[float], k: int = 4
) -> Dict:
    """Returns payload for performing hybrid search for given options.

    Args:
        query_text: The query text to search for.
        query_vector: The embedding vector (query) to search for.
        k: Number of Documents to return. Defaults to 4.

    Returns:
        dict: The payload for hybrid search.
    """
    payload = {
        "_source": {"exclude": ["vector_field"]},
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "text": {
                                "query": query_text,
                            }
                        }
                    },
                    {"knn": {"vector_field": {"vector": query_vector, "k": k}}},
                ]
            }
        },
        "size": k,
    }

    return payload


def _hybrid_search_query_with_post_filter(
    query_text: str,
    query_vector: List[float],
    k: int,
    post_filter: Dict,
) -> Dict:
    """Returns payload for performing hybrid search with post filter.

    Args:
        query_text: The query text to search for.
        query_vector: The embedding vector to search for.
        k: Number of Documents to return.
        post_filter: The post filter to apply.

    Returns:
        dict: The payload for hybrid search with post filter.
    """
    search_query = _default_hybrid_search_query(query_text, query_vector, k)

    search_query["post_filter"] = post_filter

    return search_query


class OpenSearchVectorSearch(VectorStore):
    """`Amazon OpenSearch Vector Engine` vector store.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import OpenSearchVectorSearch
            opensearch_vector_search = OpenSearchVectorSearch(
                "http://localhost:9200",
                "embeddings",
                embedding_function
            )

    """

    def __init__(
        self,
        opensearch_url: str,
        index_name: str,
        embedding_function: Embeddings,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index_name = index_name
        http_auth = kwargs.get("http_auth")
        self.is_aoss = _is_aoss_enabled(http_auth=http_auth)
        self.client = _get_opensearch_client(opensearch_url, **kwargs)
        self.async_client = _get_async_opensearch_client(opensearch_url, **kwargs)
        self.engine = kwargs.get("engine", "nmslib")
        self.bulk_size = kwargs.get("bulk_size", 500)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def __add(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        bulk_size = bulk_size if bulk_size is not None else self.bulk_size
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        dim = len(embeddings[0])
        engine = kwargs.get("engine", self.engine)
        space_type = kwargs.get("space_type", "l2")
        ef_search = kwargs.get("ef_search", 512)
        ef_construction = kwargs.get("ef_construction", 512)
        m = kwargs.get("m", 16)
        vector_field = kwargs.get("vector_field", "vector_field")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)

        _validate_aoss_with_engines(self.is_aoss, engine)

        mapping = _default_text_mapping(
            dim, engine, space_type, ef_search, ef_construction, m, vector_field
        )

        return _bulk_ingest_embeddings(
            self.client,
            index_name,
            embeddings,
            texts,
            metadatas=metadatas,
            ids=ids,
            vector_field=vector_field,
            text_field=text_field,
            mapping=mapping,
            max_chunk_bytes=max_chunk_bytes,
            is_aoss=self.is_aoss,
        )

    async def __aadd(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        bulk_size = bulk_size if bulk_size is not None else self.bulk_size
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        dim = len(embeddings[0])
        engine = kwargs.get("engine", self.engine)
        space_type = kwargs.get("space_type", "l2")
        ef_search = kwargs.get("ef_search", 512)
        ef_construction = kwargs.get("ef_construction", 512)
        m = kwargs.get("m", 16)
        vector_field = kwargs.get("vector_field", "vector_field")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)

        _validate_aoss_with_engines(self.is_aoss, engine)

        mapping = _default_text_mapping(
            dim, engine, space_type, ef_search, ef_construction, m, vector_field
        )

        return await _abulk_ingest_embeddings(
            self.async_client,
            index_name,
            embeddings,
            texts,
            metadatas=metadatas,
            ids=ids,
            vector_field=vector_field,
            text_field=text_field,
            mapping=mapping,
            max_chunk_bytes=max_chunk_bytes,
            is_aoss=self.is_aoss,
        )

    def delete_index(self, index_name: Optional[str] = None) -> Optional[bool]:
        """Deletes a given index from vectorstore."""
        if index_name is None:
            if self.index_name is None:
                raise ValueError("index_name must be provided.")
            index_name = self.index_name
        try:
            self.client.indices.delete(index=index_name)
            return True
        except Exception as e:
            raise e

    def index_exists(self, index_name: Optional[str] = None) -> Optional[bool]:
        """If given index present in vectorstore, returns True else False."""
        if index_name is None:
            if self.index_name is None:
                raise ValueError("index_name must be provided.")
            index_name = self.index_name

        return self.client.indices.exists(index=index_name)

    def create_index(
        self,
        dimension: int,
        index_name: Optional[str] = uuid.uuid4().hex,
        **kwargs: Any,
    ) -> Optional[str]:
        """Create a new Index with given arguments"""
        is_appx_search = kwargs.get("is_appx_search", True)
        vector_field = kwargs.get("vector_field", "vector_field")
        kwargs.get("text_field", "text")
        http_auth = kwargs.get("http_auth")
        is_aoss = _is_aoss_enabled(http_auth=http_auth)

        if is_aoss and not is_appx_search:
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `approximate_search`"
            )

        if is_appx_search:
            engine = kwargs.get("engine", self.engine)
            space_type = kwargs.get("space_type", "l2")
            ef_search = kwargs.get("ef_search", 512)
            ef_construction = kwargs.get("ef_construction", 512)
            m = kwargs.get("m", 16)

            _validate_aoss_with_engines(is_aoss, engine)

            mapping = _default_text_mapping(
                dimension,
                engine,
                space_type,
                ef_search,
                ef_construction,
                m,
                vector_field,
            )
        else:
            mapping = _default_scripting_text_mapping(dimension)

        if self.index_exists(index_name):
            raise RuntimeError(f"The index, {index_name} already exists.")
        self.client.indices.create(index=index_name, body=mapping)
        return index_name

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            bulk_size: Bulk API request count; Default: 500

        Returns:
            List of ids from adding the texts into the vectorstore.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        bulk_size = bulk_size if bulk_size is not None else self.bulk_size
        return self.__add(
            texts,
            embeddings,
            metadatas=metadatas,
            ids=ids,
            bulk_size=bulk_size,
            **kwargs,
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously run more texts through the embeddings
        and add to the vectorstore.
        """
        embeddings = await self.embedding_function.aembed_documents(list(texts))
        bulk_size = bulk_size if bulk_size is not None else self.bulk_size
        return await self.__aadd(
            texts,
            embeddings,
            metadatas=metadatas,
            ids=ids,
            bulk_size=bulk_size,
            **kwargs,
        )

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add the given texts and embeddings to the vectorstore.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            bulk_size: Bulk API request count; Default: 500

        Returns:
            List of ids from adding the texts into the vectorstore.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".
        """
        texts, embeddings = zip(*text_embeddings)
        bulk_size = bulk_size if bulk_size is not None else self.bulk_size
        return self.__add(
            list(texts),
            list(embeddings),
            metadatas=metadatas,
            ids=ids,
            bulk_size=bulk_size,
            **kwargs,
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        refresh_indices: Optional[bool] = True,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents from the Opensearch index.

        Args:
            ids: List of ids of documents to delete.
            refresh_indices: Whether to refresh the index
                            after deleting documents. Defaults to True.
        """
        try:
            from opensearchpy.helpers import bulk
        except ImportError:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)

        body = []

        if ids is None:
            raise ValueError("ids must be provided.")

        for _id in ids:
            body.append({"_op_type": "delete", "_index": self.index_name, "_id": _id})

        if len(body) > 0:
            try:
                bulk(self.client, body, refresh=refresh_indices, ignore_status=404)
                return True
            except Exception as e:
                raise e
        else:
            return False

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Asynchronously delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")

        actions = [{"delete": {"_index": self.index_name, "_id": id_}} for id_ in ids]
        response = await self.async_client.bulk(body=actions, **kwargs)
        return not any(
            item.get("delete", {}).get("error") for item in response["items"]
        )

    def configure_search_pipelines(
        self,
        pipeline_name: str,
        keyword_weight: float = 0.7,
        vector_weight: float = 0.3,
    ) -> dict:
        """
        Configures a search pipeline for hybrid search.
        Args:
            pipeline_name: Name of the pipeline
            keyword_weight: Weight for keyword search
            vector_weight: Weight for vector search
        Returns:
            response: Acknowledgement of the pipeline creation.
            (if there is any error while configuring the pipeline, it will return None)
        Raises:
            Exception: If an error occurs
        """
        if not pipeline_name.isidentifier():
            raise ValueError(f"Invalid pipeline name: {pipeline_name}")

        path = f"/_search/pipeline/{pipeline_name}"

        payload = {
            "description": "Post processor for hybrid search",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [keyword_weight, vector_weight]},
                        },
                    }
                }
            ],
        }

        response = self.client.transport.perform_request(
            method="PUT", url=path, body=payload
        )
        return response

    def search_pipeline_exists(self, pipeline_name: str) -> bool:
        """
        Checks if a search pipeline exists.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            bool: True if the pipeline exists, False otherwise

        Raises:
            Exception: If an error occurs

        Example:
            >>> search_pipeline_exists("my_pipeline_1")
            True
            >>> search_pipeline_exists("my_pipeline_2")
            False
        """
        if not pipeline_name.isidentifier():
            raise ValueError(f"Invalid pipeline name: {pipeline_name}")

        existed_pipelines = self.client.transport.perform_request(
            method="GET", url="/_search/pipeline/"
        )

        return pipeline_name in existed_pipelines

    def get_search_pipeline_info(self, pipeline_name: str) -> Optional[Dict]:
        """
        Get information about a search pipeline.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            dict: Information about the pipeline
            None: If pipeline does not exist

        Raises:
            Exception: If an error occurs

        Example:
            >>> get_search_pipeline_info("my_pipeline_1")
            {'search_pipeline_1': {
                "description": "Post processor for hybrid search",
                "phase_results_processors": [
                    {
                        "normalization-processor": {
                            "normalization": {"technique": "min_max"},
                            "combination": {
                                "technique": "arithmetic_mean",
                                "parameters": {"weights": [0.7, 0.3]}
                            }
                        }
                    }
                ]
            }
            }
            >>> get_search_pipeline_info("my_pipeline_2")
            None
        """
        response = None

        if not pipeline_name.isidentifier():
            raise ValueError(f"Invalid pipeline name: {pipeline_name}")

        response = self.client.transport.perform_request(
            method="GET", url=f"/_search/pipeline/{pipeline_name}"
        )

        return response

    @staticmethod
    def _identity_fn(score: float) -> float:
        return score

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        return self._identity_fn

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = 0.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            score_threshold: Specify a score threshold to return only documents
            above the threshold. Defaults to 0.0.

        Returns:
            List of Documents most similar to the query.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

            metadata_field: Document field that metadata is stored in. Defaults to
            "metadata".
            Can be set to a special value "*" to include the entire document.

        Optional Args for Approximate Search:
            search_type: "approximate_search"; default: "approximate_search"

            boolean_filter: A Boolean filter is a post filter consists of a Boolean
            query that contains a k-NN query and a filter.

            subquery_clause: Query clause on the knn vector field; default: "must"

            lucene_filter: the Lucene algorithm decides whether to perform an exact
            k-NN search with pre-filtering or an approximate search with modified
            post-filtering. (deprecated, use `efficient_filter`)

            efficient_filter: the Lucene Engine or Faiss Engine decides whether to
            perform an exact k-NN search with pre-filtering or an approximate search
            with modified post-filtering.

        Optional Args for Script Scoring Search:
            search_type: "script_scoring"; default: "approximate_search"

            space_type: "l2", "l1", "linf", "cosinesimil", "innerproduct",
            "hammingbit"; default: "l2"

            pre_filter: script_score query to pre-filter documents before identifying
            nearest neighbors; default: {"match_all": {}}

        Optional Args for Painless Scripting Search:
            search_type: "painless_scripting"; default: "approximate_search"

            space_type: "l2Squared", "l1Norm", "cosineSimilarity"; default: "l2Squared"

            pre_filter: script_score query to pre-filter documents before identifying
            nearest neighbors; default: {"match_all": {}}
        """
        docs_with_scores = self.similarity_search_with_score(
            query, k, score_threshold, **kwargs
        )
        return [doc[0] for doc in docs_with_scores]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        score_threshold: Optional[float] = 0.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to the embedding vector."""
        docs_with_scores = self.similarity_search_with_score_by_vector(
            embedding, k, score_threshold, **kwargs
        )
        return [doc[0] for doc in docs_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = 0.0,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and it's scores most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            score_threshold: Specify a score threshold to return only documents
            above the threshold. Defaults to 0.0.

        Returns:
            List of Documents along with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        # added query_text to kwargs for Hybrid Search
        kwargs["query_text"] = query
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding, k, score_threshold, **kwargs
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        score_threshold: Optional[float] = 0.0,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and it's scores most similar to the embedding vector.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            score_threshold: Specify a score threshold to return only documents
            above the threshold. Defaults to 0.0.

        Returns:
            List of Documents along with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        hits = self._raw_similarity_search_with_score_by_vector(
            embedding=embedding, k=k, score_threshold=score_threshold, **kwargs
        )

        documents_with_scores = [
            (
                Document(
                    page_content=hit["_source"][text_field],
                    metadata=(
                        hit["_source"]
                        if metadata_field == "*" or metadata_field not in hit["_source"]
                        else hit["_source"][metadata_field]
                    ),
                    id=hit["_id"],
                ),
                hit["_score"],
            )
            for hit in hits
        ]
        return documents_with_scores

    def _raw_similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        score_threshold: Optional[float] = 0.0,
        **kwargs: Any,
    ) -> List[dict]:
        """Return raw opensearch documents (dict) including vectors,
        scores most similar to the embedding vector.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            score_threshold: Specify a score threshold to return only documents
            above the threshold. Defaults to 0.0.

        Returns:
            List of dict with its scores most similar to the embedding.

        Optional Args:
            same as `similarity_search`
        """
        search_type = kwargs.get("search_type", "approximate_search")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        filter = kwargs.get("filter", {})

        if (
            self.is_aoss
            and search_type != "approximate_search"
            and search_type != SCRIPT_SCORING_SEARCH
        ):
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `approximate_search` and `script_scoring`"
            )

        if search_type == "approximate_search":
            boolean_filter = kwargs.get("boolean_filter", {})
            subquery_clause = kwargs.get("subquery_clause", "must")
            efficient_filter = kwargs.get("efficient_filter", {})
            # `lucene_filter` is deprecated, added for Backwards Compatibility
            lucene_filter = kwargs.get("lucene_filter", {})

            if boolean_filter != {} and efficient_filter != {}:
                raise ValueError(
                    "Both `boolean_filter` and `efficient_filter` are provided which "
                    "is invalid"
                )

            if lucene_filter != {} and efficient_filter != {}:
                raise ValueError(
                    "Both `lucene_filter` and `efficient_filter` are provided which "
                    "is invalid. `lucene_filter` is deprecated"
                )

            if lucene_filter != {} and boolean_filter != {}:
                raise ValueError(
                    "Both `lucene_filter` and `boolean_filter` are provided which "
                    "is invalid. `lucene_filter` is deprecated"
                )

            if (
                efficient_filter == {}
                and boolean_filter == {}
                and lucene_filter == {}
                and filter != {}
            ):
                if self.engine in ["faiss", "lucene"]:
                    efficient_filter = filter
                else:
                    boolean_filter = filter

            if boolean_filter != {}:
                search_query = _approximate_search_query_with_boolean_filter(
                    embedding,
                    boolean_filter,
                    k=k,
                    vector_field=vector_field,
                    subquery_clause=subquery_clause,
                    score_threshold=score_threshold,
                )
            elif efficient_filter != {}:
                search_query = _approximate_search_query_with_efficient_filter(
                    embedding,
                    efficient_filter,
                    k=k,
                    vector_field=vector_field,
                    score_threshold=score_threshold,
                )
            elif lucene_filter != {}:
                warnings.warn(
                    "`lucene_filter` is deprecated. Please use the keyword argument"
                    " `efficient_filter`"
                )
                search_query = _approximate_search_query_with_efficient_filter(
                    embedding,
                    lucene_filter,
                    k=k,
                    vector_field=vector_field,
                    score_threshold=score_threshold,
                )
            else:
                search_query = _default_approximate_search_query(
                    embedding,
                    k=k,
                    vector_field=vector_field,
                    score_threshold=score_threshold,
                )
        elif search_type == SCRIPT_SCORING_SEARCH:
            space_type = kwargs.get("space_type", "l2")
            pre_filter = kwargs.get("pre_filter", MATCH_ALL_QUERY)
            search_query = _default_script_query(
                embedding,
                k,
                space_type,
                pre_filter,
                vector_field,
                score_threshold=score_threshold,
            )
        elif search_type == PAINLESS_SCRIPTING_SEARCH:
            space_type = kwargs.get("space_type", "l2Squared")
            pre_filter = kwargs.get("pre_filter", MATCH_ALL_QUERY)
            search_query = _default_painless_scripting_query(
                embedding,
                k,
                space_type,
                pre_filter,
                vector_field,
                score_threshold=score_threshold,
            )

        elif search_type == HYBRID_SEARCH:
            search_pipeline = kwargs.get("search_pipeline")
            post_filter = kwargs.get("post_filter", {})
            query_text = kwargs.get("query_text")
            path = f"/{index_name}/_search?search_pipeline={search_pipeline}"

            if query_text is None:
                raise ValueError("query_text must be provided for hybrid search")

            if search_pipeline is None:
                raise ValueError("search_pipeline must be provided for hybrid search")

            # embedding the query_text
            embeded_query = self.embedding_function.embed_query(query_text)

            # if post filter is provided
            if post_filter != {}:
                # hybrid search with post filter
                payload = _hybrid_search_query_with_post_filter(
                    query_text, embeded_query, k, post_filter
                )
            else:
                # hybrid search without post filter
                payload = _default_hybrid_search_query(query_text, embeded_query, k)

            response = self.client.transport.perform_request(
                method="GET", url=path, body=payload
            )

            return [hit for hit in response["hits"]["hits"]]

        else:
            raise ValueError("Invalid `search_type` provided as an argument")

        response = self.client.search(index=index_name, body=search_query)

        return [hit for hit in response["hits"]["hits"]]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        # Get embedding of the user query
        embedding = self.embedding_function.embed_query(query)

        # Do ANN/KNN search to get top fetch_k results where fetch_k >= k
        results = self._raw_similarity_search_with_score_by_vector(
            embedding, fetch_k, **kwargs
        )

        embeddings = [result["_source"][vector_field] for result in results]

        # Rerank top k results using MMR, (mmr_selected is a list of indices)
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )

        return [
            Document(
                page_content=results[i]["_source"][text_field],
                metadata=results[i]["_source"][metadata_field],
                id=results[i]["_id"],
            )
            for i in mmr_selected
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        bulk_size: Optional[int] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> OpenSearchVectorSearch:
        """Construct OpenSearchVectorSearch wrapper from raw texts.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                opensearch_vector_search = OpenSearchVectorSearch.from_texts(
                    texts,
                    embeddings,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        embeddings = embedding.embed_documents(texts)
        bulk_size = (
            bulk_size if bulk_size is not None else getattr(cls, "bulk_size", 500)
        )
        return cls.from_embeddings(
            embeddings,
            texts,
            embedding,
            metadatas=metadatas,
            bulk_size=bulk_size,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        bulk_size: Optional[int] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> OpenSearchVectorSearch:
        """Asynchronously construct OpenSearchVectorSearch wrapper from raw texts.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                opensearch_vector_search = await OpenSearchVectorSearch.afrom_texts(
                    texts,
                    embeddings,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        embeddings = await embedding.aembed_documents(texts)
        bulk_size = (
            bulk_size if bulk_size is not None else getattr(cls, "bulk_size", 500)
        )
        return await cls.afrom_embeddings(
            embeddings,
            texts,
            embedding,
            metadatas=metadatas,
            bulk_size=bulk_size,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[List[float]],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        bulk_size: Optional[int] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> OpenSearchVectorSearch:
        """Construct OpenSearchVectorSearch wrapper from pre-vectorized embeddings.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embedder = OpenAIEmbeddings()
                embeddings = embedder.embed_documents(["foo", "bar"])
                opensearch_vector_search = OpenSearchVectorSearch.from_embeddings(
                    embeddings,
                    texts,
                    embedder,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        opensearch_url = get_from_dict_or_env(
            kwargs, "opensearch_url", "OPENSEARCH_URL"
        )
        # List of arguments that needs to be removed from kwargs
        # before passing kwargs to get opensearch client
        keys_list = [
            "opensearch_url",
            "index_name",
            "is_appx_search",
            "vector_field",
            "text_field",
            "engine",
            "space_type",
            "ef_search",
            "ef_construction",
            "m",
            "max_chunk_bytes",
            "is_aoss",
        ]
        bulk_size = (
            bulk_size if bulk_size is not None else getattr(cls, "bulk_size", 500)
        )
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        dim = len(embeddings[0])
        # Get the index name from either from kwargs or ENV Variable
        # before falling back to random generation
        index_name = get_from_dict_or_env(
            kwargs, "index_name", "OPENSEARCH_INDEX_NAME", default=uuid.uuid4().hex
        )
        is_appx_search = kwargs.get("is_appx_search", True)
        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)
        http_auth = kwargs.get("http_auth")
        is_aoss = _is_aoss_enabled(http_auth=http_auth)
        engine = None

        if is_aoss and not is_appx_search:
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `approximate_search`"
            )

        if is_appx_search:
            engine = kwargs.get("engine", "nmslib")
            space_type = kwargs.get("space_type", "l2")
            ef_search = kwargs.get("ef_search", 512)
            ef_construction = kwargs.get("ef_construction", 512)
            m = kwargs.get("m", 16)

            _validate_aoss_with_engines(is_aoss, engine)

            mapping = _default_text_mapping(
                dim, engine, space_type, ef_search, ef_construction, m, vector_field
            )
        else:
            mapping = _default_scripting_text_mapping(dim)

        [kwargs.pop(key, None) for key in keys_list]
        client = _get_opensearch_client(opensearch_url, **kwargs)
        _bulk_ingest_embeddings(
            client,
            index_name,
            embeddings,
            texts,
            ids=ids,
            metadatas=metadatas,
            vector_field=vector_field,
            text_field=text_field,
            mapping=mapping,
            max_chunk_bytes=max_chunk_bytes,
            is_aoss=is_aoss,
        )
        kwargs["engine"] = engine
        return cls(opensearch_url, index_name, embedding, **kwargs)

    @classmethod
    async def afrom_embeddings(
        cls,
        embeddings: List[List[float]],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        bulk_size: Optional[int] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> OpenSearchVectorSearch:
        """Asynchronously construct OpenSearchVectorSearch wrapper from pre-vectorized
        embeddings.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embedder = OpenAIEmbeddings()
                embeddings = await embedder.aembed_documents(["foo", "bar"])
                opensearch_vector_search =
                    await OpenSearchVectorSearch.afrom_embeddings(
                        embeddings,
                        texts,
                        embedder,
                        opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        opensearch_url = get_from_dict_or_env(
            kwargs, "opensearch_url", "OPENSEARCH_URL"
        )
        # List of arguments that needs to be removed from kwargs
        # before passing kwargs to get opensearch client
        keys_list = [
            "opensearch_url",
            "index_name",
            "is_appx_search",
            "vector_field",
            "text_field",
            "engine",
            "space_type",
            "ef_search",
            "ef_construction",
            "m",
            "max_chunk_bytes",
            "is_aoss",
        ]
        bulk_size = (
            bulk_size if bulk_size is not None else getattr(cls, "bulk_size", 500)
        )
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        dim = len(embeddings[0])
        # Get the index name from either from kwargs or ENV Variable
        # before falling back to random generation
        index_name = get_from_dict_or_env(
            kwargs, "index_name", "OPENSEARCH_INDEX_NAME", default=uuid.uuid4().hex
        )
        is_appx_search = kwargs.get("is_appx_search", True)
        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)
        http_auth = kwargs.get("http_auth")
        is_aoss = _is_aoss_enabled(http_auth=http_auth)
        engine = None

        if is_aoss and not is_appx_search:
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `approximate_search`"
            )

        if is_appx_search:
            engine = kwargs.get("engine", "nmslib")
            space_type = kwargs.get("space_type", "l2")
            ef_search = kwargs.get("ef_search", 512)
            ef_construction = kwargs.get("ef_construction", 512)
            m = kwargs.get("m", 16)

            _validate_aoss_with_engines(is_aoss, engine)

            mapping = _default_text_mapping(
                dim, engine, space_type, ef_search, ef_construction, m, vector_field
            )
        else:
            mapping = _default_scripting_text_mapping(dim)

        [kwargs.pop(key, None) for key in keys_list]
        client = _get_async_opensearch_client(opensearch_url, **kwargs)
        await _abulk_ingest_embeddings(
            client,
            index_name,
            embeddings,
            texts,
            ids=ids,
            metadatas=metadatas,
            vector_field=vector_field,
            text_field=text_field,
            mapping=mapping,
            max_chunk_bytes=max_chunk_bytes,
            is_aoss=is_aoss,
        )
        kwargs["engine"] = engine
        return cls(opensearch_url, index_name, embedding, **kwargs)
