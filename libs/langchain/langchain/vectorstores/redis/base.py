"""Wrapper around Redis vector database."""

from __future__ import annotations

import logging
import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import yaml

from langchain._api import deprecated
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utilities.redis import (
    _array_to_buffer,
    check_redis_module_exist,
    get_client,
)
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain.vectorstores.redis.constants import (
    REDIS_REQUIRED_MODULES,
    REDIS_TAG_SEPARATOR,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.query import Query

    from langchain.vectorstores.redis.filters import RedisFilterExpression
    from langchain.vectorstores.redis.schema import RedisModel


def _redis_key(prefix: str) -> str:
    """Redis key schema for a given prefix."""
    return f"{prefix}:{uuid.uuid4().hex}"


def _redis_prefix(index_name: str) -> str:
    """Redis key prefix for a given index."""
    return f"doc:{index_name}"


def _default_relevance_score(val: float) -> float:
    return 1 - val


def check_index_exists(client: RedisType, index_name: str) -> bool:
    """Check if Redis index exists."""
    try:
        client.ft(index_name).info()
    except:  # noqa: E722
        logger.info("Index does not exist")
        return False
    logger.info("Index already exists")
    return True


class Redis(VectorStore):
    """Wrapper around Redis vector database.

    To use, you should have the ``redis`` python package installed
    and have a running Redis Enterprise or Redis-Stack server

    For production use cases, it is recommended to use Redis Enterprise
    as the scaling, performance, stability and availability is much
    better than Redis-Stack.

    For testing and prototyping, however, this is not required.
    Redis-Stack is available as a docker container the full vector
    search API available.

    .. code-block:: bash
        # to run redis stack in docker locally
        docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

    Once running, you can connect to the redis server with the following url schemas:
    - redis://<host>:<port> # simple connection
    - redis://<username>:<password>@<host>:<port> # connection with authentication
    - rediss://<host>:<port> # connection with SSL
    - rediss://<username>:<password>@<host>:<port> # connection with SSL and auth


    Examples:

    The following examples show various ways to use the Redis VectorStore with
    LangChain.

    For all the following examples assume we have the following imports:

    .. code-block:: python

        from langchain.vectorstores import Redis
        from langchain.embeddings import OpenAIEmbeddings

    Initialize, create index, and load Documents
        .. code-block:: python

            from langchain.vectorstores import Redis
            from langchain.embeddings import OpenAIEmbeddings

            rds = Redis.from_documents(
                documents, # a list of Document objects from loaders or created
                embeddings, # an Embeddings object
                redis_url="redis://localhost:6379",
            )

    Initialize, create index, and load Documents with metadata
        .. code-block:: python


            rds = Redis.from_texts(
                texts, # a list of strings
                metadata, # a list of metadata dicts
                embeddings, # an Embeddings object
                redis_url="redis://localhost:6379",
            )

    Initialize, create index, and load Documents with metadata and return keys

        .. code-block:: python

            rds, keys = Redis.from_texts_return_keys(
                texts, # a list of strings
                metadata, # a list of metadata dicts
                embeddings, # an Embeddings object
                redis_url="redis://localhost:6379",
            )

    For use cases where the index needs to stay alive, you can initialize
    with an index name such that it's easier to reference later

        .. code-block:: python

            rds = Redis.from_texts(
                texts, # a list of strings
                metadata, # a list of metadata dicts
                embeddings, # an Embeddings object
                index_name="my-index",
                redis_url="redis://localhost:6379",
            )

    Initialize and connect to an existing index (from above)

        .. code-block:: python

            rds = Redis.from_existing_index(
                embeddings, # an Embeddings object
                index_name="my-index",
                redis_url="redis://localhost:6379",
            )


    Advanced examples:

    Custom vector schema can be supplied to change the way that
    Redis creates the underlying vector schema. This is useful
    for production use cases where you want to optimize the
    vector schema for your use case. ex. using HNSW instead of
    FLAT (knn) which is the default

        .. code-block:: python

            vector_schema = {
                "algorithm": "HNSW"
            }

            rds = Redis.from_texts(
                texts, # a list of strings
                metadata, # a list of metadata dicts
                embeddings, # an Embeddings object
                vector_schema=vector_schema,
                redis_url="redis://localhost:6379",
            )

    Custom index schema can be supplied to change the way that the
    metadata is indexed. This is useful for you would like to use the
    hybrid querying (filtering) capability of Redis.

    By default, this implementation will automatically generate the index
    schema according to the following rules:
        - All strings are indexed as text fields
        - All numbers are indexed as numeric fields
        - All lists of strings are indexed as tag fields (joined by
            langchain.vectorstores.redis.constants.REDIS_TAG_SEPARATOR)
        - All None values are not indexed but still stored in Redis these are
            not retrievable through the interface here, but the raw Redis client
            can be used to retrieve them.
        - All other types are not indexed

    To override these rules, you can pass in a custom index schema like the following

        .. code-block:: yaml

            tag:
                - name: credit_score
            text:
                - name: user
                - name: job

    Typically, the ``credit_score`` field would be a text field since it's a string,
    however, we can override this behavior by specifying the field type as shown with
    the yaml config (can also be a dictionary) above and the code below.

        .. code-block:: python

            rds = Redis.from_texts(
                texts, # a list of strings
                metadata, # a list of metadata dicts
                embeddings, # an Embeddings object
                index_schema="path/to/index_schema.yaml", # can also be a dictionary
                redis_url="redis://localhost:6379",
            )

    When connecting to an existing index where a custom schema has been applied, it's
    important to pass in the same schema to the ``from_existing_index`` method.
    Otherwise, the schema for newly added samples will be incorrect and metadata
    will not be returned.

    """

    DEFAULT_VECTOR_SCHEMA = {
        "name": "content_vector",
        "algorithm": "FLAT",
        "dims": 1536,
        "distance_metric": "COSINE",
        "datatype": "FLOAT32",
    }

    def __init__(
        self,
        redis_url: str,
        index_name: str,
        embedding: Embeddings,
        index_schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        vector_schema: Optional[Dict[str, Union[str, int]]] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self._check_deprecated_kwargs(kwargs)
        try:
            # TODO use importlib to check if redis is installed
            import redis  # noqa: F401

        except ImportError as e:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            ) from e

        self.index_name = index_name
        self._embeddings = embedding
        try:
            redis_client = get_client(redis_url=redis_url, **kwargs)
            # check if redis has redisearch module installed
            check_redis_module_exist(redis_client, REDIS_REQUIRED_MODULES)
        except ValueError as e:
            raise ValueError(f"Redis failed to connect: {e}")

        self.client = redis_client
        self.relevance_score_fn = relevance_score_fn
        self._schema = self._get_schema_with_defaults(index_schema, vector_schema)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embeddings

    @classmethod
    def from_texts_return_keys(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        index_schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        vector_schema: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs: Any,
    ) -> Tuple[Redis, List[str]]:
        """Create a Redis vectorstore from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new Redis index if it doesn't already exist
            3. Adds the documents to the newly created Redis index.
            4. Returns the keys of the newly created documents once stored.

        This method will generate schema based on the metadata passed in
        if the `index_schema` is not defined. If the `index_schema` is defined,
        it will compare against the generated schema and warn if there are
        differences. If you are purposefully defining the schema for the
        metadata, then you can ignore that warning.

        To examine the schema options, initialize an instance of this class
        and print out the schema using the `Redis.schema`` property. This
        will include the content and content_vector classes which are
        always present in the langchain schema.

        Example:
            .. code-block:: python

                from langchain.vectorstores import Redis
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redis, keys = Redis.from_texts_return_keys(
                    texts,
                    embeddings,
                    redis_url="redis://localhost:6379"
                )

        Args:
            texts (List[str]): List of texts to add to the vectorstore.
            embedding (Embeddings): Embeddings to use for the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadata
                dicts to add to the vectorstore. Defaults to None.
            index_name (Optional[str], optional): Optional name of the index to
                create or add to. Defaults to None.
            index_schema (Optional[Union[Dict[str, str], str, os.PathLike]], optional):
                Optional fields to index within the metadata. Overrides generated
                schema. Defaults to None.
            vector_schema (Optional[Dict[str, Union[str, int]]], optional): Optional
                vector schema to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the Redis client.

        Returns:
            Tuple[Redis, List[str]]: Tuple of the Redis instance and the keys of
                the newly created documents.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
        """
        try:
            # TODO use importlib to check if redis is installed
            import redis  # noqa: F401

            from langchain.vectorstores.redis.schema import read_schema

        except ImportError as e:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            ) from e

        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if "redis_url" in kwargs:
            kwargs.pop("redis_url")

        # flag to use generated schema
        if "generate" in kwargs:
            kwargs.pop("generate")

        # Name of the search index if not given
        if not index_name:
            index_name = uuid.uuid4().hex

        # type check for metadata
        if metadatas:
            if isinstance(metadatas, list) and len(metadatas) != len(texts):  # type: ignore  # noqa: E501
                raise ValueError("Number of metadatas must match number of texts")
            if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
                raise ValueError("Metadatas must be a list of dicts")

            generated_schema = _generate_field_schema(metadatas[0])
            if index_schema:
                # read in the schema solely to compare to the generated schema
                user_schema = read_schema(index_schema)

                # the very rare case where a super user decides to pass the index
                # schema and a document loader is used that has metadata which
                # we need to map into fields.
                if user_schema != generated_schema:
                    logger.warning(
                        "`index_schema` does not match generated metadata schema.\n"
                        + "If you meant to manually override the schema, please "
                        + "ignore this message.\n"
                        + f"index_schema: {user_schema}\n"
                        + f"generated_schema: {generated_schema}\n"
                    )
            else:
                # use the generated schema
                index_schema = generated_schema

        # Create instance
        instance = cls(
            redis_url,
            index_name,
            embedding,
            index_schema=index_schema,
            vector_schema=vector_schema,
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
        index_schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        vector_schema: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs: Any,
    ) -> Redis:
        """Create a Redis vectorstore from a list of texts.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new Redis index if it doesn't already exist
            3. Adds the documents to the newly created Redis index.

        This method will generate schema based on the metadata passed in
        if the `index_schema` is not defined. If the `index_schema` is defined,
        it will compare against the generated schema and warn if there are
        differences. If you are purposefully defining the schema for the
        metadata, then you can ignore that warning.

        To examine the schema options, initialize an instance of this class
        and print out the schema using the `Redis.schema`` property. This
        will include the content and content_vector classes which are
        always present in the langchain schema.


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

        Args:
            texts (List[str]): List of texts to add to the vectorstore.
            embedding (Embeddings): Embedding model class (i.e. OpenAIEmbeddings)
                for embedding queries.
            metadatas (Optional[List[dict]], optional): Optional list of metadata dicts
                to add to the vectorstore. Defaults to None.
            index_name (Optional[str], optional): Optional name of the index to create
                or add to. Defaults to None.
            index_schema (Optional[Union[Dict[str, str], str, os.PathLike]], optional):
                Optional fields to index within the metadata. Overrides generated
                schema. Defaults to None.
            vector_schema (Optional[Dict[str, Union[str, int]]], optional): Optional
                vector schema to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the Redis client.

        Returns:
            Redis: Redis VectorStore instance.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ImportError: If the redis python package is not installed.
        """
        instance, _ = cls.from_texts_return_keys(
            texts,
            embedding,
            metadatas=metadatas,
            index_name=index_name,
            index_schema=index_schema,
            vector_schema=vector_schema,
            **kwargs,
        )
        return instance

    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str,
        schema: Union[Dict[str, str], str, os.PathLike],
        **kwargs: Any,
    ) -> Redis:
        """Connect to an existing Redis index.

        Example:
            .. code-block:: python

                from langchain.vectorstores import Redis
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch = Redis.from_existing_index(
                    embeddings,
                    index_name="my-index",
                    redis_url="redis://username:password@localhost:6379"
                )

        Args:
            embedding (Embeddings): Embedding model class (i.e. OpenAIEmbeddings)
                for embedding queries.
            index_name (str): Name of the index to connect to.
            schema (Union[Dict[str, str], str, os.PathLike]): Schema of the index
                and the vector schema. Can be a dict, or path to yaml file

            **kwargs (Any): Additional keyword arguments to pass to the Redis client.

        Returns:
            Redis: Redis VectorStore instance.

        Raises:
            ValueError: If the index does not exist.
            ImportError: If the redis python package is not installed.
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = get_client(redis_url=redis_url, **kwargs)
            # check if redis has redisearch module installed
            check_redis_module_exist(client, REDIS_REQUIRED_MODULES)
            # ensure that the index already exists
            assert check_index_exists(
                client, index_name
            ), f"Index {index_name} does not exist"
        except Exception as e:
            raise ValueError(f"Redis failed to connect: {e}")

        return cls(
            redis_url,
            index_name,
            embedding,
            index_schema=schema,
            **kwargs,
        )

    @property
    def schema(self) -> Dict[str, List[Any]]:
        """Return the schema of the index."""
        return self._schema.as_dict()

    def write_schema(self, path: Union[str, os.PathLike]) -> None:
        """Write the schema to a yaml file."""
        with open(path, "w+") as f:
            yaml.dump(self.schema, f)

    @staticmethod
    def delete(
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a Redis entry.

        Args:
            ids: List of ids (keys in redis) to delete.
            redis_url: Redis connection url. This should be passed in the kwargs
                or set as an environment variable: REDIS_URL.

        Returns:
            bool: Whether or not the deletions were successful.

        Raises:
            ValueError: If the redis python package is not installed.
            ValueError: If the ids (keys in redis) are not provided
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

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 1000,
        clean_metadata: bool = True,
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
            if isinstance(metadatas, list) and len(metadatas) != len(texts):  # type: ignore  # noqa: E501
                raise ValueError("Number of metadatas must match number of texts")
            if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
                raise ValueError("Metadatas must be a list of dicts")

        # Write data to redis
        pipeline = self.client.pipeline(transaction=False)
        for i, text in enumerate(texts):
            # Use provided values by default or fallback
            key = keys_or_ids[i] if keys_or_ids else _redis_key(prefix)
            metadata = metadatas[i] if metadatas else {}
            metadata = _prepare_metadata(metadata) if clean_metadata else metadata
            embedding = (
                embeddings[i] if embeddings else self._embeddings.embed_query(text)
            )
            pipeline.hset(
                key,
                mapping={
                    self._schema.content_key: text,
                    self._schema.content_vector_key: _array_to_buffer(
                        embedding, self._schema.vector_dtype
                    ),
                    **metadata,
                },
            )
            ids.append(key)

            # Write batch
            if i % batch_size == 0:
                pipeline.execute()

        # Cleanup final batch
        pipeline.execute()
        return ids

    def as_retriever(self, **kwargs: Any) -> RedisVectorStoreRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return RedisVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)

    @deprecated("0.0.272", alternative="similarity_search(distance_threshold=0.1)")
    def similarity_search_limit_score(
        self, query: str, k: int = 4, score_threshold: float = 0.2, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text within the
        score_threshold range.

        Deprecated: Use similarity_search with distance_threshold instead.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_threshold (float): The minimum matching *distance* required
                for a document to be considered a match. Defaults to 0.2.

        Returns:
            List[Document]: A list of documents that are most similar to the query text
                including the match score for each document.

        Note:
            If there are no documents that satisfy the score_threshold value,
            an empty list is returned.

        """
        return self.similarity_search(
            query, k=k, distance_threshold=score_threshold, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[RedisFilterExpression] = None,
        return_metadata: bool = True,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with **vector distance**.

        The "scores" returned from this function are the raw vector
        distances from the query vector. For similarity scores, use
        ``similarity_search_with_relevance_scores``.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.

        Returns:
            List[Tuple[Document, float]]: A list of documents that are
                most similar to the query with the distance for each document.
        """
        try:
            import redis

        except ImportError as e:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            ) from e

        if "score_threshold" in kwargs:
            logger.warning(
                "score_threshold is deprecated. Use distance_threshold instead."
                + "score_threshold should only be used in "
                + "similarity_search_with_relevance_scores."
                + "score_threshold will be removed in a future release.",
            )

        redis_query, params_dict = self._prepare_query(
            query,
            k=k,
            filter=filter,
            with_metadata=return_metadata,
            with_distance=True,
            **kwargs,
        )

        # Perform vector search
        # ignore type because redis-py is wrong about bytes
        try:
            results = self.client.ft(self.index_name).search(redis_query, params_dict)  # type: ignore  # noqa: E501
        except redis.exceptions.ResponseError as e:
            # split error message and see if it starts with "Syntax"
            if str(e).split(" ")[0] == "Syntax":
                raise ValueError(
                    "Query failed with syntax error. "
                    + "This is likely due to malformation of "
                    + "filter, vector, or query argument"
                ) from e
            raise e

        # Prepare document results
        docs_with_scores: List[Tuple[Document, float]] = []
        for result in results.docs:
            metadata = {}
            if return_metadata:
                metadata = {"id": result.id}
                metadata.update(self._collect_metadata(result))

            doc = Document(page_content=result.content, metadata=metadata)
            distance = self._calculate_fp_distance(result.distance)
            docs_with_scores.append((doc, distance))

        return docs_with_scores

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[RedisFilterExpression] = None,
        return_metadata: bool = True,
        distance_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.
            distance_threshold (Optional[float], optional): Distance threshold
                for vector distance from query vector. Defaults to None.

        Returns:
            List[Document]: A list of documents that are most similar to the query
                text.

        """
        try:
            import redis

        except ImportError as e:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            ) from e

        if "score_threshold" in kwargs:
            logger.warning(
                "score_threshold is deprecated. Use distance_threshold instead."
                + "score_threshold should only be used in "
                + "similarity_search_with_relevance_scores."
                + "score_threshold will be removed in a future release.",
            )

        redis_query, params_dict = self._prepare_query(
            query,
            k=k,
            filter=filter,
            distance_threshold=distance_threshold,
            with_metadata=return_metadata,
            with_distance=False,
        )

        # Perform vector search
        # ignore type because redis-py is wrong about bytes
        try:
            results = self.client.ft(self.index_name).search(redis_query, params_dict)  # type: ignore  # noqa: E501
        except redis.exceptions.ResponseError as e:
            # split error message and see if it starts with "Syntax"
            if str(e).split(" ")[0] == "Syntax":
                raise ValueError(
                    "Query failed with syntax error. "
                    + "This is likely due to malformation of "
                    + "filter, vector, or query argument"
                ) from e
            raise e

        # Prepare document results
        docs = []
        for result in results.docs:
            metadata = {}
            if return_metadata:
                metadata = {"id": result.id}
                metadata.update(self._collect_metadata(result))

            content_key = self._schema.content_key
            docs.append(
                Document(page_content=getattr(result, content_key), metadata=metadata)
            )
        return docs

    def _collect_metadata(self, result: "Document") -> Dict[str, Any]:
        """Collect metadata from Redis.

        Method ensures that there isn't a mismatch between the metadata
        and the index schema passed to this class by the user or generated
        by this class.

        Args:
            result (Document): redis.commands.search.Document object returned
                from Redis.

        Returns:
            Dict[str, Any]: Collected metadata.
        """
        # new metadata dict as modified by this method
        meta = {}
        for key in self._schema.metadata_keys:
            try:
                meta[key] = getattr(result, key)
            except AttributeError:
                # warning about attribute missing
                logger.warning(
                    f"Metadata key {key} not found in metadata. "
                    + "Setting to None. \n"
                    + "Metadata fields defined for this instance: "
                    + f"{self._schema.metadata_keys}"
                )
                meta[key] = None
        return meta

    def _prepare_query(
        self,
        query: str,
        k: int = 4,
        filter: Optional[RedisFilterExpression] = None,
        distance_threshold: Optional[float] = None,
        with_metadata: bool = True,
        with_distance: bool = False,
    ) -> Tuple["Query", Dict[str, Any]]:
        # Creates embedding vector from user query
        embedding = self._embeddings.embed_query(query)

        # Creates Redis query
        params_dict: Dict[str, Union[str, bytes, float]] = {
            "vector": _array_to_buffer(embedding, self._schema.vector_dtype),
        }

        # prepare return fields including score
        return_fields = [self._schema.content_key]
        if with_distance:
            return_fields.append("distance")
        if with_metadata:
            return_fields.extend(self._schema.metadata_keys)

        if distance_threshold:
            params_dict["distance_threshold"] = distance_threshold
            return (
                self._prepare_range_query(
                    k, filter=filter, return_fields=return_fields
                ),
                params_dict,
            )
        return (
            self._prepare_vector_query(k, filter=filter, return_fields=return_fields),
            params_dict,
        )

    def _prepare_range_query(
        self,
        k: int,
        filter: Optional[RedisFilterExpression] = None,
        return_fields: Optional[List[str]] = None,
    ) -> "Query":
        try:
            from redis.commands.search.query import Query
        except ImportError as e:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            ) from e
        return_fields = return_fields or []
        vector_key = self._schema.content_vector_key
        base_query = f"@{vector_key}:[VECTOR_RANGE $distance_threshold $vector]"

        if filter:
            base_query = "(" + base_query + " " + str(filter) + ")"

        query_string = base_query + "=>{$yield_distance_as: distance}"

        return (
            Query(query_string)
            .return_fields(*return_fields)
            .sort_by("distance")
            .paging(0, k)
            .dialect(2)
        )

    def _prepare_vector_query(
        self,
        k: int,
        filter: Optional[RedisFilterExpression] = None,
        return_fields: Optional[List[str]] = None,
    ) -> "Query":
        """Prepare query for vector search.

        Args:
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            query: Query object.
        """
        try:
            from redis.commands.search.query import Query
        except ImportError as e:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            ) from e
        return_fields = return_fields or []
        query_prefix = "*"
        if filter:
            query_prefix = f"{str(filter)}"
        vector_key = self._schema.content_vector_key
        base_query = f"({query_prefix})=>[KNN {k} @{vector_key} $vector AS distance]"

        query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("distance")
            .paging(0, k)
            .dialect(2)
        )
        return query

    def _get_schema_with_defaults(
        self,
        index_schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        vector_schema: Optional[Dict[str, Union[str, int]]] = None,
    ) -> "RedisModel":
        # should only be called after init of Redis (so Import handled)
        from langchain.vectorstores.redis.schema import RedisModel, read_schema

        schema = RedisModel()
        # read in schema (yaml file or dict) and
        # pass to the Pydantic validators
        if index_schema:
            schema_values = read_schema(index_schema)
            schema = RedisModel(**schema_values)

            # ensure user did not exclude the content field
            # no modifications if content field found
            schema.add_content_field()

        # if no content_vector field, add vector field to schema
        # this makes adding a vector field to the schema optional when
        # the user just wants additional metadata
        try:
            # see if user overrode the content vector
            schema.content_vector
            # if user overrode the content vector, check if they
            # also passed vector schema. This won't be used since
            # the index schema overrode the content vector
            if vector_schema:
                logger.warning(
                    "`vector_schema` is ignored since content_vector is "
                    + "overridden in `index_schema`."
                )

        # user did not override content vector
        except ValueError:
            # set default vector schema and update with user provided schema
            # if the user provided any
            vector_field = self.DEFAULT_VECTOR_SCHEMA.copy()
            if vector_schema:
                vector_field.update(vector_schema)

            # add the vector field either way
            schema.add_vector_field(vector_field)
        return schema

    def _create_index(self, dim: int = 1536) -> None:
        try:
            from redis.commands.search.indexDefinition import (  # type: ignore
                IndexDefinition,
                IndexType,
            )

        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        # Set vector dimension
        # can't obtain beforehand because we don't
        # know which embedding model is being used.
        self._schema.content_vector.dims = dim

        # Check if index exists
        if not check_index_exists(self.client, self.index_name):
            prefix = _redis_prefix(self.index_name)

            # Create Redis Index
            self.client.ft(self.index_name).create_index(
                fields=self._schema.get_fields(),
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )

    def _calculate_fp_distance(self, distance: str) -> float:
        """Calculate the distance based on the vector datatype

        Two datatypes supported:
        - FLOAT32
        - FLOAT64

        if it's FLOAT32, we need to round the distance to 4 decimal places
        otherwise, round to 7 decimal places.
        """
        if self._schema.content_vector.datatype == "FLOAT32":
            return round(float(distance), 4)
        return round(float(distance), 7)

    def _check_deprecated_kwargs(self, kwargs: Mapping[str, Any]) -> None:
        """Check for deprecated kwargs."""

        deprecated_kwargs = {
            "redis_host": "redis_url",
            "redis_port": "redis_url",
            "redis_password": "redis_url",
            "content_key": "index_schema",
            "vector_key": "vector_schema",
            "distance_metric": "vector_schema",
        }
        for key, value in kwargs.items():
            if key in deprecated_kwargs:
                raise ValueError(
                    f"Keyword argument '{key}' is deprecated. "
                    f"Please use '{deprecated_kwargs[key]}' instead."
                )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.relevance_score_fn:
            return self.relevance_score_fn

        metric_map = {
            "COSINE": self._cosine_relevance_score_fn,
            "IP": self._max_inner_product_relevance_score_fn,
            "L2": self._euclidean_relevance_score_fn,
        }
        try:
            return metric_map[self._schema.content_vector.distance_metric]
        except KeyError:
            return _default_relevance_score


def _generate_field_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a schema for the search index in Redis based on the input metadata.

    Given a dictionary of metadata, this function categorizes each metadata
        field into one of the three categories:
    - text: The field contains textual data.
    - numeric: The field contains numeric data (either integer or float).
    - tag: The field contains list of tags (strings).

    Args
        data (Dict[str, Any]): A dictionary where keys are metadata field names
            and values are the metadata values.

    Returns:
        Dict[str, Any]: A dictionary with three keys "text", "numeric", and "tag".
            Each key maps to a list of fields that belong to that category.

    Raises:
        ValueError: If a metadata field cannot be categorized into any of
            the three known types.
    """
    result: Dict[str, Any] = {
        "text": [],
        "numeric": [],
        "tag": [],
    }

    for key, value in data.items():
        # Numeric fields
        try:
            int(value)
            result["numeric"].append({"name": key})
            continue
        except (ValueError, TypeError):
            pass

        # None values are not indexed as of now
        if value is None:
            continue

        # if it's a list of strings, we assume it's a tag
        if isinstance(value, (list, tuple)):
            if not value or isinstance(value[0], str):
                result["tag"].append({"name": key})
            else:
                name = type(value[0]).__name__
                raise ValueError(
                    f"List/tuple values should contain strings: '{key}': {name}"
                )
            continue

        # Check if value is string before processing further
        if isinstance(value, str):
            result["text"].append({"name": key})
            continue

        # Unable to classify the field value
        name = type(value).__name__
        raise ValueError(
            "Could not generate Redis index field type mapping "
            + f"for metadata: '{key}': {name}"
        )

    return result


def _prepare_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare metadata for indexing in Redis by sanitizing its values.

    - String, integer, and float values remain unchanged.
    - None or empty values are replaced with empty strings.
    - Lists/tuples of strings are joined into a single string with a comma separator.

    Args:
        metadata (Dict[str, Any]): A dictionary where keys are metadata
            field names and values are the metadata values.

    Returns:
        Dict[str, Any]: A sanitized dictionary ready for indexing in Redis.

    Raises:
        ValueError: If any metadata value is not one of the known
            types (string, int, float, or list of strings).
    """

    def raise_error(key: str, value: Any) -> None:
        raise ValueError(
            f"Metadata value for key '{key}' must be a string, int, "
            + f"float, or list of strings. Got {type(value).__name__}"
        )

    clean_meta: Dict[str, Union[str, float, int]] = {}
    for key, value in metadata.items():
        if not value:
            clean_meta[key] = ""
            continue

        # No transformation needed
        if isinstance(value, (str, int, float)):
            clean_meta[key] = value

        # if it's a list/tuple of strings, we join it
        elif isinstance(value, (list, tuple)):
            if not value or isinstance(value[0], str):
                clean_meta[key] = REDIS_TAG_SEPARATOR.join(value)
            else:
                raise_error(key, value)
        else:
            raise_error(key, value)
    return clean_meta


class RedisVectorStoreRetriever(VectorStoreRetriever):
    """Retriever for Redis VectorStore."""

    vectorstore: Redis
    """Redis VectorStore."""
    search_type: str = "similarity"
    """Type of search to perform. Can be either
    'similarity',
    'similarity_distance_threshold',
    'similarity_score_threshold'
    """

    search_kwargs: Dict[str, Any] = {
        "k": 4,
        "score_threshold": 0.9,
        # set to None to avoid distance used in score_threshold search
        "distance_threshold": None,
    }
    """Default search kwargs."""

    allowed_search_types = [
        "similarity",
        "similarity_distance_threshold",
        "similarity_score_threshold",
    ]
    """Allowed search types."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        elif self.search_type == "similarity_distance_threshold":
            if self.search_kwargs["distance_threshold"] is None:
                raise ValueError(
                    "distance_threshold must be provided for "
                    + "similarity_distance_threshold retriever"
                )
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
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
