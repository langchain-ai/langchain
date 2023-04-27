"""Wrapper around OpenSearch vector database."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)
SCRIPT_SCORING_SEARCH = "script_scoring"
PAINLESS_SCRIPTING_SEARCH = "painless_scripting"
MATCH_ALL_QUERY = {"match_all": {}}  # type: Dict


def _import_opensearch() -> Any:
    """Import OpenSearch if available, otherwise raise error."""
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        raise ValueError(IMPORT_OPENSEARCH_PY_ERROR)
    return OpenSearch


def _import_bulk() -> Any:
    """Import bulk if available, otherwise raise error."""
    try:
        from opensearchpy.helpers import bulk
    except ImportError:
        raise ValueError(IMPORT_OPENSEARCH_PY_ERROR)
    return bulk


def _import_not_found_error() -> Any:
    """Import not found error if available, otherwise raise error."""
    try:
        from opensearchpy.exceptions import NotFoundError
    except ImportError:
        raise ValueError(IMPORT_OPENSEARCH_PY_ERROR)
    return NotFoundError


def _get_opensearch_client(opensearch_url: str, **kwargs: Any) -> Any:
    """Get OpenSearch client from the opensearch_url, otherwise raise error."""
    try:
        opensearch = _import_opensearch()
        client = opensearch(opensearch_url, **kwargs)
    except ValueError as e:
        raise ValueError(
            f"OpenSearch client string provided is not in proper format. "
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


def _bulk_ingest_embeddings(
    client: Any,
    index_name: str,
    embeddings: List[List[float]],
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    vector_field: str = "vector_field",
    text_field: str = "text",
    mapping: Dict = {},
) -> List[str]:
    """Bulk Ingest Embeddings into given index."""
    bulk = _import_bulk()
    not_found_error = _import_not_found_error()
    requests = []
    ids = []
    mapping = mapping

    try:
        client.indices.get(index=index_name)
    except not_found_error:
        client.indices.create(index=index_name, body=mapping)

    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas else {}
        _id = str(uuid.uuid4())
        request = {
            "_op_type": "index",
            "_index": index_name,
            vector_field: embeddings[i],
            text_field: text,
            "metadata": metadata,
            "_id": _id,
        }
        requests.append(request)
        ids.append(_id)
    bulk(client, requests)
    client.indices.refresh(index=index_name)
    return ids


def _default_scripting_text_mapping(
    dim: int,
    vector_field: str = "vector_field",
) -> Dict:
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
) -> Dict:
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
    size: int = 4,
    k: int = 4,
    vector_field: str = "vector_field",
) -> Dict:
    """For Approximate k-NN Search, this is the default query."""
    return {
        "size": size,
        "query": {"knn": {vector_field: {"vector": query_vector, "k": k}}},
    }


def _approximate_search_query_with_boolean_filter(
    query_vector: List[float],
    boolean_filter: Dict,
    size: int = 4,
    k: int = 4,
    vector_field: str = "vector_field",
    subquery_clause: str = "must",
) -> Dict:
    """For Approximate k-NN Search, with Boolean Filter."""
    return {
        "size": size,
        "query": {
            "bool": {
                "filter": boolean_filter,
                subquery_clause: [
                    {"knn": {vector_field: {"vector": query_vector, "k": k}}}
                ],
            }
        },
    }


def _approximate_search_query_with_lucene_filter(
    query_vector: List[float],
    lucene_filter: Dict,
    size: int = 4,
    k: int = 4,
    vector_field: str = "vector_field",
) -> Dict:
    """For Approximate k-NN Search, with Lucene Filter."""
    search_query = _default_approximate_search_query(
        query_vector, size, k, vector_field
    )
    search_query["query"]["knn"][vector_field]["filter"] = lucene_filter
    return search_query


def _default_script_query(
    query_vector: List[float],
    space_type: str = "l2",
    pre_filter: Dict = MATCH_ALL_QUERY,
    vector_field: str = "vector_field",
) -> Dict:
    """For Script Scoring Search, this is the default query."""
    return {
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
        }
    }


def __get_painless_scripting_source(
    space_type: str, query_vector: List[float], vector_field: str = "vector_field"
) -> str:
    """For Painless Scripting, it returns the script source based on space type."""
    source_value = (
        "(1.0 + "
        + space_type
        + "("
        + str(query_vector)
        + ", doc['"
        + vector_field
        + "']))"
    )
    if space_type == "cosineSimilarity":
        return source_value
    else:
        return "1/" + source_value


def _default_painless_scripting_query(
    query_vector: List[float],
    space_type: str = "l2Squared",
    pre_filter: Dict = MATCH_ALL_QUERY,
    vector_field: str = "vector_field",
) -> Dict:
    """For Painless Scripting Search, this is the default query."""
    source = __get_painless_scripting_source(space_type, query_vector)
    return {
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
        }
    }


def _get_kwargs_value(kwargs: Any, key: str, default_value: Any) -> Any:
    """Get the value of the key if present. Else get the default_value."""
    if key in kwargs:
        return kwargs.get(key)
    return default_value


class OpenSearchVectorSearch(VectorStore):
    """Wrapper around OpenSearch as a vector database.

    Example:
        .. code-block:: python

            from langchain import OpenSearchVectorSearch
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
        self.client = _get_opensearch_client(opensearch_url, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
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
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        text_field = _get_kwargs_value(kwargs, "text_field", "text")
        dim = len(embeddings[0])
        engine = _get_kwargs_value(kwargs, "engine", "nmslib")
        space_type = _get_kwargs_value(kwargs, "space_type", "l2")
        ef_search = _get_kwargs_value(kwargs, "ef_search", 512)
        ef_construction = _get_kwargs_value(kwargs, "ef_construction", 512)
        m = _get_kwargs_value(kwargs, "m", 16)
        vector_field = _get_kwargs_value(kwargs, "vector_field", "vector_field")

        mapping = _default_text_mapping(
            dim, engine, space_type, ef_search, ef_construction, m, vector_field
        )

        return _bulk_ingest_embeddings(
            self.client,
            self.index_name,
            embeddings,
            texts,
            metadatas,
            vector_field,
            text_field,
            mapping,
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        By default supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

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

            size: number of results the query actually returns; default: 4

            boolean_filter: A Boolean filter consists of a Boolean query that
            contains a k-NN query and a filter.

            subquery_clause: Query clause on the knn vector field; default: "must"

            lucene_filter: the Lucene algorithm decides whether to perform an exact
            k-NN search with pre-filtering or an approximate search with modified
            post-filtering.

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
        embedding = self.embedding_function.embed_query(query)
        search_type = _get_kwargs_value(kwargs, "search_type", "approximate_search")
        text_field = _get_kwargs_value(kwargs, "text_field", "text")
        metadata_field = _get_kwargs_value(kwargs, "metadata_field", "metadata")
        vector_field = _get_kwargs_value(kwargs, "vector_field", "vector_field")

        if search_type == "approximate_search":
            size = _get_kwargs_value(kwargs, "size", 4)
            boolean_filter = _get_kwargs_value(kwargs, "boolean_filter", {})
            subquery_clause = _get_kwargs_value(kwargs, "subquery_clause", "must")
            lucene_filter = _get_kwargs_value(kwargs, "lucene_filter", {})
            if boolean_filter != {} and lucene_filter != {}:
                raise ValueError(
                    "Both `boolean_filter` and `lucene_filter` are provided which "
                    "is invalid"
                )
            if boolean_filter != {}:
                search_query = _approximate_search_query_with_boolean_filter(
                    embedding, boolean_filter, size, k, vector_field, subquery_clause
                )
            elif lucene_filter != {}:
                search_query = _approximate_search_query_with_lucene_filter(
                    embedding, lucene_filter, size, k, vector_field
                )
            else:
                search_query = _default_approximate_search_query(
                    embedding, size, k, vector_field
                )
        elif search_type == SCRIPT_SCORING_SEARCH:
            space_type = _get_kwargs_value(kwargs, "space_type", "l2")
            pre_filter = _get_kwargs_value(kwargs, "pre_filter", MATCH_ALL_QUERY)
            search_query = _default_script_query(
                embedding, space_type, pre_filter, vector_field
            )
        elif search_type == PAINLESS_SCRIPTING_SEARCH:
            space_type = _get_kwargs_value(kwargs, "space_type", "l2Squared")
            pre_filter = _get_kwargs_value(kwargs, "pre_filter", MATCH_ALL_QUERY)
            search_query = _default_painless_scripting_query(
                embedding, space_type, pre_filter, vector_field
            )
        else:
            raise ValueError("Invalid `search_type` provided as an argument")

        response = self.client.search(index=self.index_name, body=search_query)
        hits = [hit["_source"] for hit in response["hits"]["hits"][:k]]
        documents = [
            Document(
                page_content=hit[text_field],
                metadata=hit
                if metadata_field == "*" or metadata_field not in hit
                else hit[metadata_field],
            )
            for hit in hits
        ]
        return documents

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> OpenSearchVectorSearch:
        """Construct OpenSearchVectorSearch wrapper from raw documents.

        Example:
            .. code-block:: python

                from langchain import OpenSearchVectorSearch
                from langchain.embeddings import OpenAIEmbeddings
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
        ]
        embeddings = embedding.embed_documents(texts)
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        dim = len(embeddings[0])
        # Get the index name from either from kwargs or ENV Variable
        # before falling back to random generation
        index_name = get_from_dict_or_env(
            kwargs, "index_name", "OPENSEARCH_INDEX_NAME", default=uuid.uuid4().hex
        )
        is_appx_search = _get_kwargs_value(kwargs, "is_appx_search", True)
        vector_field = _get_kwargs_value(kwargs, "vector_field", "vector_field")
        text_field = _get_kwargs_value(kwargs, "text_field", "text")
        if is_appx_search:
            engine = _get_kwargs_value(kwargs, "engine", "nmslib")
            space_type = _get_kwargs_value(kwargs, "space_type", "l2")
            ef_search = _get_kwargs_value(kwargs, "ef_search", 512)
            ef_construction = _get_kwargs_value(kwargs, "ef_construction", 512)
            m = _get_kwargs_value(kwargs, "m", 16)

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
            metadatas,
            vector_field,
            text_field,
            mapping,
        )
        return cls(opensearch_url, index_name, embedding, **kwargs)
