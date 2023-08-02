"""Wrapper around Redis vector database."""

from __future__ import annotations
import re
import json
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    Pattern,
    Union
)

import numpy as np
<<<<<<< HEAD
=======
import numbers
from pydantic import root_validator
>>>>>>> 053ed66a (Add support to Numeric and Tags)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)


from pydantic import BaseModel, Field, validator
from redis.commands.search.field import (
    GeoField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from typing_extensions import Literal
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import root_validator
from langchain.utilities.redis import get_client
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.query import Query


# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20600},
    {"name": "searchlight", "ver": 20600},
]

# distance mmetrics
REDIS_DISTANCE_METRICS = Literal["COSINE", "IP", "L2"]

class TokenEscaper:
    """
    Escape punctuation within an input string.
    """

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/]"

    def __init__(self, escape_chars_re: Optional[Pattern] = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        def escape_symbol(match):
            value = match.group(0)
            return f"\\{value}"

        return self.escaped_chars_re.sub(escape_symbol, value)


def _check_redis_module_exist(client: RedisType, required_modules: List[dict]) -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in required_modules:
        if module["name"] in installed_modules and int(
            installed_modules[module["name"]][b"ver"]
        ) >= int(module["ver"]):
            return
    # otherwise raise error
    error_message = (
        "Redis cannot be used as a vector database without RediSearch >=2.4"
        "Please head to https://redis.io/docs/stack/search/quick_start/"
        "to know more about installing the RediSearch module within Redis Stack."
    )
    logger.error(error_message)
    raise ValueError(error_message)


def _check_index_exists(client: RedisType, index_name: str) -> bool:
    """Check if Redis index exists."""
    try:
        client.ft(index_name).info()
    except:  # noqa: E722
        logger.info("Index does not exist")
        return False
    logger.info("Index already exists")
    return True


def _redis_key(prefix: str) -> str:
    """Redis key schema for a given prefix."""
    return f"{prefix}:{uuid.uuid4().hex}"


def _redis_prefix(index_name: str) -> str:
    """Redis key prefix for a given index."""
    return f"doc:{index_name}"


def _default_relevance_score(val: float) -> float:
    return 1 - val


def array_to_buffer(array: List[float], dtype: Any = np.float32) -> bytes:
    return np.array(array).astype(dtype).tobytes()


class BaseField(BaseModel):
    name: str = Field(...)
    sortable: Optional[bool] = False


class TextFieldSchema(BaseField):
    weight: Optional[float] = 1
    no_stem: Optional[bool] = False
    phonetic_matcher: Optional[str] = None
    withsuffixtrie: Optional[bool] = False

    def as_field(self):
        return TextField(
            self.name,
            weight=self.weight,
            no_stem=self.no_stem,
            phonetic_matcher=self.phonetic_matcher,
            sortable=self.sortable,
        )


class TagFieldSchema(BaseField):
    separator: Optional[str] = ","
    case_sensitive: Optional[bool] = False

    def as_field(self):
        return TagField(
            self.name,
            separator=self.separator,
            case_sensitive=self.case_sensitive,
            sortable=self.sortable,
        )


class NumericFieldSchema(BaseField):
    def as_field(self):
        return NumericField(self.name, sortable=self.sortable)


class GeoFieldSchema(BaseField):
    def as_field(self):
        return GeoField(self.name, sortable=self.sortable)


class BaseVectorField(BaseModel):
    name: str = Field(...)
    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: str = Field(default="COSINE")
    initial_cap: int = Field(default=20000)

    @validator("algorithm", "datatype", "distance_metric")
    @classmethod
    def uppercase_strings(cls, v):
        return v.upper()

    @property
    def metric(self):
        return self.distance_metric

class FlatVectorField(BaseVectorField):
    algorithm: object = Literal["FLAT"]
    block_size: int = Field(default=1000)

    def as_field(self):
        return VectorField(
            self.name,
            self.algorithm,
            {
                "TYPE": self.datatype,
                "DIM": self.dims,
                "DISTANCE_METRIC": self.distance_metric,
                "INITIAL_CAP": self.initial_cap,
                "BLOCK_SIZE": self.block_size,
            },
        )


class HNSWVectorField(BaseVectorField):
    algorithm: object = Literal["HNSW"]
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.8)

    def as_field(self):
        return VectorField(
            self.name,
            self.algorithm,
            {
                "TYPE": self.datatype,
                "DIM": self.dims,
                "DISTANCE_METRIC": self.distance_metric,
                "INITIAL_CAP": self.initial_cap,
                "M": self.m,
                "EF_CONSTRUCTION": self.ef_construction,
                "EF_RUNTIME": self.ef_runtime,
                "EPSILON": self.epsilon,
            },
        )

class RedisMetadata(BaseModel):
    tag: Optional[List[TagFieldSchema]] = None
    text: Optional[List[TextFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    geo: Optional[List[GeoFieldSchema]] = None
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]] = None

    def get_fields(self):
        redis_fields = []
        for field_name in self.fields.model_fields.keys():
            field_group = getattr(self.fields, field_name)
            if field_group is not None:
                for field in field_group:
                    redis_fields.append(field.as_field())
        return redis_fields

    @property
    def keys(self):
        keys = []
        for field_name in self.fields.model_fields.keys():
            field_group = getattr(self.fields, field_name)
            if field_group is not None:
                for field in field_group:
                    keys.append(field.name)
        return keys

class RedisFilter:
    escaper = TokenEscaper()

    def __init__(self, field):
        self._field = field
        self._filters = []

    def __str__(self):
        base = self.to_string()
        if self._filters:
            base += "".join(self._filters)
        return base

    def __iadd__(self, other) -> "RedisFilter":
        "intersection '+='"
        self._filters.append(f" {other.to_string()}")
        return self

    def __iand__(self, other) -> "RedisFilter":
        "union '&='"
        self._filters.append(f" | {other.to_string()}")
        return self

    def __isub__(self, other) -> "RedisFilter":
        "subtract '-='"
        self._filters.append(f" -{other.to_string()}")
        return self

    def __ixor__(self, other) -> "RedisFilter":
        "With optional '^='"
        self._filters.append(f" ~{other.to_string()}")
        return self

    def to_string(self) -> str:
        raise NotImplementedError


class TagFilter(RedisFilter):
    def __init__(self, field, tags: List[str]):
        super().__init__(field)
        self.tags = tags

    def to_string(self) -> str:
        """Converts the tag filter to a string.

        Returns:
            str: The tag filter as a string.
        """
        if not isinstance(self.tags, list):
            self.tags = [self.tags]
        return (
            "@"
            + self._field
            + ":{"
            + " | ".join([self.escaper.escape(tag) for tag in self.tags])
            + "}"
        )

class NumericFilter(RedisFilter):
    def __init__(self, field, minval, maxval, min_exclusive=False, max_exclusive=False):
        """Filter for Numeric fields.

        Args:
            field (str): The field to filter on.
            minval (int): The minimum value.
            maxval (int): The maximum value.
            min_exclusive (bool, optional): Whether the minimum value is exclusive. Defaults to False.
            max_exclusive (bool, optional): Whether the maximum value is exclusive. Defaults to False.
        """
        self.top = maxval if not max_exclusive else f"({maxval}"
        self.bottom = minval if not min_exclusive else f"({minval}"
        super().__init__(field)

    def to_string(self):
        return "@" + self._field + ":[" + str(self.bottom) + " " + str(self.top) + "]"


class TextFilter(RedisFilter):
    def __init__(self, field, text: str):
        """Filter for Text fields.
        Args:
            field (str): The field to filter on.
            text (str): The text to filter on.
        """
        super().__init__(field)
        self.text = text

    def to_string(self) -> str:
        """Converts the filter to a string.

        Returns:
            str: The filter as a string.
        """
        return "@" + self._field + ":" + self.escaper.escape(self.text)


class Redis(VectorStore):
    """Wrapper around Redis vector database.

    To use, you should have the ``redis`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import Redis
            from langchain.embeddings import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vectorstore = Redis(
                redis_url="redis://username:password@localhost:6379"
                index_name="my-index",
                embedding_function=embeddings.embed_query,
            )

    To use a redis replication setup with multiple redis server and redis sentinels
    set "redis_url" to "redis+sentinel://" scheme. With this url format a path is
    needed holding the name of the redis service within the sentinels to get the
    correct redis server connection. The default service name is "mymaster".

    An optional username or password is used for booth connections to the rediserver
    and the sentinel, different passwords for server and sentinel are not supported.
    And as another constraint only one sentinel instance can be given:

    Example:
        .. code-block:: python

            vectorstore = Redis(
                redis_url="redis+sentinel://username:password@sentinelhost:26379/mymaster/0"
                index_name="my-index",
                embedding_function=embeddings.embed_query,
            )
    """

    _default_content_vector_schema = {
        "name": "content_vector",
        "dims": 1536,
        "distance_metric": "cosine",
        "algorithm": "flat",
        "datatype": "float32"
    }

    def __init__(
        self,
        redis_url: str,
        index_name: str,
        embedding_function: Callable,
        content_key: str = "content",
        vector_key: str = "content_vector",
        vector_schema: Optional[Dict] = None,
        metadata_schema: Optional[Dict] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self.index_name = index_name
        self.content_key = content_key
        self.vector_key = vector_key
        self.embedding_function = embedding_function
        try:
            redis_client = get_client(redis_url=redis_url, **kwargs)
            # check if redis has redisearch module installed
            _check_redis_module_exist(redis_client, REDIS_REQUIRED_MODULES)
        except ValueError as e:
            raise ValueError(f"Redis failed to connect: {e}")

        self.client = redis_client
        self.relevance_score_fn = relevance_score_fn
        self._vector_schema = self._default_content_vector_schema
        if isinstance(vector_schema, dict):
            self._vector_schema.update(vector_schema)
        if metadata_schema:
            self._metadata_schema = RedisMetadata(**metadata_schema)
            self._metadata_keys = self._metadata_schema.keys
        else:
            self._metadata_schema = None
            self._metadata_keys = []

        self._distance_metric = self._vector_schema["distance_metric"].upper()
        self._select_relevance_score_fn()


    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.relevance_score_fn:
            return self.relevance_score_fn

        metric_map = {
            "COSINE": self._cosine_relevance_score_fn,
            "IP": self._max_inner_product_relevance_score_fn,
            "L2": self._euclidean_relevance_score_fn
        }
        if self._distance_metric in metric_map:
            return metric_map[self._distance_metric]
        else:
            return _default_relevance_score

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Accept embedding object directly
        return None

    def _create_index(self, dim: int = 1536) -> None:

        fields = []
        try:
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType

            # Field for content
            fields.append(TextField(self.content_key))

            # Field for Content Vector
            self._vector_schema["dims"] = dim
            if self._vector_schema["algorithm"].upper() == "FLAT":
                fields.append(FlatVectorField(**self._vector_schema).as_field())
            else:
                fields.append(HNSWVectorField(**self._vector_schema).as_field())

            # Fields for metadata
            if self._metadata_schema:
                fields.extend(self._metadata_schema.get_fields())

        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        # Check if index exists
        if not _check_index_exists(self.client, self.index_name):
            prefix = _redis_prefix(self.index_name)

            # Create Redis Index
            self.client.ft(self.index_name).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.
            keys (List[str]) or ids (List[str]): Identifiers of entries.
                Defaults to None.
            batch_size (int, optional): Batch size to use for writes. Defaults to 1000.

        Returns:
            List[str]: List of ids added to the vectorstore
        """
        ids = []
        prefix = _redis_prefix(self.index_name)

        # Get keys or ids from kwargs
        # Other vectorstores use ids
        keys_or_ids = kwargs.get("keys", kwargs.get("ids"))

        # type check for metadata
        if metadatas:
            if len(metadatas) != len(texts):
                raise ValueError("Number of metadatas must match number of texts")
            if not isinstance(metadatas, list) and not isinstance(metadatas[0], dict):
                raise ValueError("Metadatas must be a list of dicts")


        # Write data to redis
        pipeline = self.client.pipeline(transaction=False)
        for i, text in enumerate(texts):
            # Use provided values by default or fallback
            key = keys_or_ids[i] if keys_or_ids else _redis_key(prefix)
            metadata = metadatas[i] if metadatas else {}
            embedding = embeddings[i] if embeddings else self.embedding_function(text)
            pipeline.hset(
                key,
                mapping={
                    self.content_key: text,
                    self.vector_key: array_to_buffer(embedding),
                    **metadata
                },
            )
            ids.append(key)

            # Write batch
            if i % batch_size == 0:
                pipeline.execute()

        # Cleanup final batch
        pipeline.execute()
        return ids

    def _similarity_search(
        self,
        query: str,
        k: int = 4,
        meta_filter: Optional[RedisFilter] = None,
        score_threshold: Optional[float] = None,
        return_score: bool = False,
        **kwargs: Any
        ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """

        # Creates embedding vector from user query
        embedding = self.embedding_function(query)

        # Creates Redis query
        params_dict: Mapping[str, Union[str, bytes]] = {
            "vector": array_to_buffer(embedding),
        }

        if score_threshold is None:
            redis_query = self._prepare_vector_query(k, meta_filter=meta_filter)
        else:
            redis_query = self._prepare_range_query(k, meta_filter=meta_filter)
            params_dict["score_threshold"] = str(score_threshold)


        # Perform vector search
        print(redis_query.query_string())
        results = self.client.ft(self.index_name).search(redis_query, params_dict)
        print(results)
        # Prepare document results
        docs = []
        scores = []
        for result in results.docs:
            metadata = {k :getattr(result, k) for k in self._metadata_keys}
            metadata["id"] = result.id
            doc = Document(page_content=result.content, metadata=metadata)
            docs.append(doc)
            if return_score:
                scores.append(result.score)
        if return_score:
            return list(zip(docs, scores))
        else:
            return docs

    def similarity_search_limit_score(
        self, query: str, k: int = 4, score_threshold: float = 0.2, meta_filter: Optional[List[RedisFilter]] = None, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text within the
        score_threshold range.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_threshold (float): The minimum matching score required for a document
                to be considered a match. Defaults to 0.2.
                Because the similarity calculation algorithm is based on cosine
                similarity, the smaller the angle, the higher the similarity.

        Returns:
            List[Document]: A list of documents that are most similar to the query text,
                including the match score for each document.

        Note:
            If there are no documents that satisfy the score_threshold value,
            an empty list is returned.

        """
        docs = self._similarity_search(
            query,
            k=k,
            score_threshold=score_threshold,
            meta_filter=meta_filter,
            return_scores=False
        )
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        meta_filter: Optional[List[RedisFilter]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        docs = self._similarity_search(
            query,
            k=k,
            meta_filter=meta_filter,
            return_scores=True
        )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        meta_filter: Optional[List[RedisFilter]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Run similarity search."""
        docs = self._similarity_search(
            query,
            k=k,
            meta_filter=meta_filter,
            return_scores=False
        )
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        raise NotImplementedError

    def _prepare_range_query(self, k: int, meta_filter: Optional[RedisFilter] = None) -> "Query":
        try:
            from redis.commands.search.query import Query
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        base_query = f'@{self.vector_key}:[VECTOR_RANGE $score_threshold $vector]'

        if meta_filter:
                base_query = "(" + base_query + " " + str(meta_filter) + ")"

        query_string = base_query + '=>{$yield_distance_as: vector_score}'
        return_fields = [*self._metadata_keys, self.content_key, "vector_score", "id"]
        return (
            Query(query_string)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )

    def _prepare_vector_query(self, k: int, meta_filter: Optional[RedisFilter] = None) -> "Query":
        try:
            from redis.commands.search.query import Query
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        query_prefix = "*"
        if meta_filter:
            query_prefix = f"{str(meta_filter)}"
        base_query = f"({query_prefix})=>[KNN {k} @{self.vector_key} $vector AS vector_score]"
        return_fields = [*self._metadata_keys, self.content_key, "vector_score", "id"]
        query = Query(base_query).return_fields(*return_fields).sort_by("vector_score").dialect(2)
        return query



    @classmethod
    def from_texts_return_keys(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        content_key: str = "content",
        vector_key: str = "content_vector",
        vector_schema: Optional[Dict[str, str]] = None,
        metadata_schema: Optional[Dict[str, List[Dict[str, str]]]] = None,
        **kwargs: Any,
    ) -> Tuple[Redis, List[str]]:
        """Create a Redis vectorstore from raw documents.
        This is a user-friendly interface that:
        1. Embeds documents.
        2. Creates a new index for the embeddings in Redis.
        3. Adds the documents to the newly created Redis index.
        4. Returns the keys of the newly created documents.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain.vectorstores import Redis
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch, keys = RediSearch.from_texts_return_keys(
                    texts,
                    embeddings,
                    redis_url="redis://username:password@localhost:6379"
                )
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if "redis_url" in kwargs:
            kwargs.pop("redis_url")

        # Name of the search index if not given
        if not index_name:
            index_name = uuid.uuid4().hex

        # Create instance
        instance = cls(
            redis_url,
            index_name,
            embedding.embed_query,
            content_key=content_key,
            vector_key=vector_key,
            vector_schema=vector_schema,
            metadata_schema=metadata_schema,
            **kwargs,
        )

        # Create embeddings over documents
        embeddings = embedding.embed_documents(texts)

        # Create the search index
        instance._create_index(dim=len(embeddings[0]))

        # Add data to Redis
        keys = instance.add_texts(texts, metadatas, embeddings)
        return instance, keys

    @classmethod
    def from_texts(
        cls: Type[Redis],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        content_key: str = "content",
        vector_key: str = "content_vector",
        vector_schema: Optional[Dict[str, str]] = None,
        metadata_schema: Optional[Dict[str, List[Dict[str, str]]]] = None,
        **kwargs: Any,
    ) -> Redis:
        """Create a Redis vectorstore from raw documents.
        This is a user-friendly interface that:
        1. Embeds documents.
        2. Creates a new index for the embeddings in Redis.
        3. Adds the documents to the newly created Redis index.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain.vectorstores import Redis
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch = RediSearch.from_texts(
                    texts,
                    embeddings,
                    redis_url="redis://username:password@localhost:6379"
                )
        """
        instance, _ = cls.from_texts_return_keys(
            texts,
            embedding,
            metadatas=metadatas,
            index_name=index_name,
            content_key=content_key,
            vector_key=vector_key,
            vector_schema=vector_schema,
            metadata_schema=metadata_schema,
            **kwargs,
        )
        return instance

    @staticmethod
    def delete(
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a Redis entry.

        Args:
            ids: List of ids (keys) to delete.

        Returns:
            bool: Whether or not the deletions were successful.
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if ids is None:
            raise ValueError("'ids' (keys)() were not provided.")

        try:
            import redis  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = get_client(redis_url=redis_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        # Check if index exists
        try:
            client.delete(*ids)
            logger.info("Entries deleted")
            return True
        except:  # noqa: E722
            # ids does not exist
            return False

    @staticmethod
    def drop_index(
        index_name: str,
        delete_documents: bool,
        **kwargs: Any,
    ) -> bool:
        """
        Drop a Redis search index.

        Args:
            index_name (str): Name of the index to drop.
            delete_documents (bool): Whether to drop the associated documents.

        Returns:
            bool: Whether or not the drop was successful.
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        try:
            import redis  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = get_client(redis_url=redis_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        # Check if index exists
        try:
            client.ft(index_name).dropindex(delete_documents)
            logger.info("Drop index")
            return True
        except:  # noqa: E722
            # Index not exist
            return False

    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str,
        content_key: str = "content",
        vector_key: str = "content_vector",
        metadata_schema: Optional[Dict[str, List[Dict[str, str]]]] = None,
        **kwargs: Any,
    ) -> Redis:
        """Connect to an existing Redis index."""
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        try:
            import redis  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = get_client(redis_url=redis_url, **kwargs)
            # check if redis has redisearch module installed
            _check_redis_module_exist(client, REDIS_REQUIRED_MODULES)
            # ensure that the index already exists
            assert _check_index_exists(
                client, index_name
            ), f"Index {index_name} does not exist"
        except Exception as e:
            raise ValueError(f"Redis failed to connect: {e}")

        return cls(
            redis_url,
            index_name,
            embedding.embed_query,
            content_key=content_key,
            vector_key=vector_key,
            metadata_schema=metadata_schema,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> RedisVectorStoreRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return RedisVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)




class RedisVectorStoreRetriever(VectorStoreRetriever):
    """Retriever for Redis VectorStore."""

    vectorstore: Redis
    """Redis VectorStore."""
    search_type: str = "similarity"
    """Type of search to perform. Can be either 'similarity' or 'similarity_limit'."""
    k: int = 4
    """Number of documents to return."""
    score_threshold: float = 0.4
    """Score threshold for similarity_limit search."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "similarity_limit"):
                raise ValueError(f"search_type of {search_type} not allowed.")
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, k=self.k)
        elif self.search_type == "similarity_limit":
            docs = self.vectorstore.similarity_search_limit_score(
                query, k=self.k, score_threshold=self.score_threshold
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("RedisVectorStoreRetriever does not support async")

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        return await self.vectorstore.aadd_documents(documents, **kwargs)
