from __future__ import annotations

import array
import functools
import hashlib
import json
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
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    from oracledb import Connection

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(
                "Failed due to a DB issue: {}".format(db_err)
            ) from db_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError("Validation failed: {}".format(val_err)) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception("An unexpected error occurred: {}".format(e))
            raise RuntimeError("Unexpected error: {}".format(e)) from e

    return cast(T, wrapper)


def _table_exists(client: Connection, table_name: str) -> bool:
    try:
        import oracledb
    except ImportError as e:
        raise ImportError(
            "Unable to import oracledb, please install with "
            "`pip install -U oracledb`."
        ) from e

    try:
        with client.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


@_handle_exceptions
def _index_exists(client: Connection, index_name: str) -> bool:
    # Check if the index exists
    query = """
        SELECT index_name 
        FROM all_indexes 
        WHERE upper(index_name) = upper(:idx_name)
        """

    with client.cursor() as cursor:
        # Execute the query
        cursor.execute(query, idx_name=index_name.upper())
        result = cursor.fetchone()

        # Check if the index exists
    return result is not None


def _get_distance_function(distance_strategy: DistanceStrategy) -> str:
    # Dictionary to map distance strategies to their corresponding function
    # names
    distance_strategy2function = {
        DistanceStrategy.EUCLIDEAN_DISTANCE: "EUCLIDEAN",
        DistanceStrategy.DOT_PRODUCT: "DOT",
        DistanceStrategy.COSINE: "COSINE",
    }

    # Attempt to return the corresponding distance function
    if distance_strategy in distance_strategy2function:
        return distance_strategy2function[distance_strategy]

    # If it's an unsupported distance strategy, raise an error
    raise ValueError(f"Unsupported distance strategy: {distance_strategy}")


def _get_index_name(base_name: str) -> str:
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f"{base_name}_{unique_id}"


@_handle_exceptions
def _create_table(client: Connection, table_name: str, embedding_dim: int) -> None:
    cols_dict = {
        "id": "RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
        "text": "CLOB",
        "metadata": "CLOB",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
    }

    if not _table_exists(client, table_name):
        with client.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
            )
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            cursor.execute(ddl)
        logger.info("Table created successfully...")
    else:
        logger.info("Table already exists...")


@_handle_exceptions
def create_index(
    client: Connection,
    vector_store: OracleVS,
    params: Optional[dict[str, Any]] = None,
) -> None:
    if params:
        if params["idx_type"] == "HNSW":
            _create_hnsw_index(
                client, vector_store.table_name, vector_store.distance_strategy, params
            )
        elif params["idx_type"] == "IVF":
            _create_ivf_index(
                client, vector_store.table_name, vector_store.distance_strategy, params
            )
        else:
            _create_hnsw_index(
                client, vector_store.table_name, vector_store.distance_strategy, params
            )
    else:
        _create_hnsw_index(
            client, vector_store.table_name, vector_store.distance_strategy, params
        )
    return


@_handle_exceptions
def _create_hnsw_index(
    client: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    defaults = {
        "idx_name": "HNSW",
        "idx_type": "HNSW",
        "neighbors": 32,
        "efConstruction": 200,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # Ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(
                        str(defaults[compulsory_key])
                    )
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # Validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults

    # Base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"create vector index {idx_name} on {table_name}(embedding) "
        f"ORGANIZATION INMEMORY NEIGHBOR GRAPH"
    )

    # Optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "neighbors" in config and "efConstruction" in config:
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )
    elif "neighbors" in config and "efConstruction" not in config:
        config["efConstruction"] = defaults["efConstruction"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )
    elif "neighbors" not in config and "efConstruction" in config:
        config["neighbors"] = defaults["neighbors"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )

    # Always included part for parallel
    parallel_part = " parallel {parallel}"

    # Combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # Format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    # Check if the index exists
    if not _index_exists(client, config["idx_name"]):
        with client.cursor() as cursor:
            cursor.execute(ddl)
            logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")


@_handle_exceptions
def _create_ivf_index(
    client: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    # Default configuration
    defaults = {
        "idx_name": "IVF",
        "idx_type": "IVF",
        "neighbor_part": 32,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # Ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(
                        str(defaults[compulsory_key])
                    )
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # Validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults

    # Base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"CREATE VECTOR INDEX {idx_name} ON {table_name}(embedding) "
        f"ORGANIZATION NEIGHBOR PARTITIONS"
    )

    # Optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "idx_type" in config and "neighbor_part" in config:
        parameters_part = (
            f" PARAMETERS (type {config['idx_type']}, neighbor"
            f" partitions {config['neighbor_part']})"
        )

    # Always included part for parallel
    parallel_part = f" PARALLEL {config['parallel']}"

    # Combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # Format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    # Check if the index exists
    if not _index_exists(client, config["idx_name"]):
        with client.cursor() as cursor:
            cursor.execute(ddl)
        logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")


@_handle_exceptions
def drop_table_purge(client: Connection, table_name: str) -> None:
    if _table_exists(client, table_name):
        cursor = client.cursor()
        with cursor:
            ddl = f"DROP TABLE {table_name} PURGE"
            cursor.execute(ddl)
        logger.info("Table dropped successfully...")
    else:
        logger.info("Table not found...")
    return


@_handle_exceptions
def drop_index_if_exists(client: Connection, index_name: str) -> None:
    if _index_exists(client, index_name):
        drop_query = f"DROP INDEX {index_name}"
        with client.cursor() as cursor:
            cursor.execute(drop_query)
            logger.info(f"Index {index_name} has been dropped.")
    else:
        logger.exception(f"Index {index_name} does not exist.")
    return


class OracleVS(VectorStore):
    """`OracleVS` vector store.

    To use, you should have both:
    - the ``oracledb`` python package installed
    - a connection string associated with a OracleDBCluster having deployed an
       Search index

    Example:
        .. code-block:: python

            from langchain.vectorstores import OracleVS
            from langchain.embeddings.openai import OpenAIEmbeddings
            import oracledb

            with oracledb.connect(user = user, passwd = pwd, dsn = dsn) as
            connection:
                print ("Database version:", connection.version)
                embeddings = OpenAIEmbeddings()
                query = ""
                vectors = OracleVS(connection, table_name, embeddings, query)
    """

    def __init__(
        self,
        client: Connection,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        table_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: Optional[str] = "What is a Oracle database",
        params: Optional[Dict[str, Any]] = None,
    ):
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        try:
            """Initialize with oracledb client."""
            self.client = client
            """Initialize with necessary components."""
            if not isinstance(embedding_function, Embeddings):
                logger.warning(
                    "`embedding_function` is expected to be an Embeddings "
                    "object, support "
                    "for passing in a function will soon be removed."
                )
            self.embedding_function = embedding_function
            self.query = query
            embedding_dim = self.get_embedding_dimension()

            self.table_name = table_name
            self.distance_strategy = distance_strategy
            self.params = params

            _create_table(client, table_name, embedding_dim)
        except oracledb.DatabaseError as db_err:
            logger.exception(f"Database error occurred while create table: {db_err}")
            raise RuntimeError(
                "Failed to create table due to a database error."
            ) from db_err
        except ValueError as val_err:
            logger.exception(f"Validation error: {val_err}")
            raise RuntimeError(
                "Failed to create table due to a validation error."
            ) from val_err
        except Exception as ex:
            logger.exception("An unexpected error occurred while creating the index.")
            raise RuntimeError(
                "Failed to create table due to an unexpected error."
            ) from ex

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """
        A property that returns an Embeddings instance embedding_function
        is an instance of Embeddings, otherwise returns None.

        Returns:
            Optional[Embeddings]: The embedding function if it's an instance of
            Embeddings, otherwise None.
        """
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    def get_embedding_dimension(self) -> int:
        # Embed the single document by wrapping it in a list
        embedded_document = self._embed_documents(
            [self.query if self.query is not None else ""]
        )

        # Get the first (and only) embedding's dimension
        return len(embedded_document[0])

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_documents(texts)
        elif callable(self.embedding_function):
            return [self.embedding_function(text) for text in texts]
        else:
            raise TypeError(
                "The embedding_function is neither Embeddings nor callable."
            )

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_query(text)
        else:
            return self.embedding_function(text)

    @_handle_exceptions
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore index.
        Args:
          texts: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the texts.
          ids: Optional list of ids for the texts that are being added to
          the vector store.
          kwargs: vectorstore specific parameters
        """

        texts = list(texts)
        if ids:
            # If ids are provided, hash them to maintain consistency
            processed_ids = [
                hashlib.sha256(_id.encode()).hexdigest()[:16].upper() for _id in ids
            ]
        elif metadatas and all("id" in metadata for metadata in metadatas):
            # If no ids are provided but metadatas with ids are, generate
            # ids from metadatas
            processed_ids = [
                hashlib.sha256(metadata["id"].encode()).hexdigest()[:16].upper()
                for metadata in metadatas
            ]
        else:
            # Generate new ids if none are provided
            generated_ids = [
                str(uuid.uuid4()) for _ in texts
            ]  # uuid4 is more standard for random UUIDs
            processed_ids = [
                hashlib.sha256(_id.encode()).hexdigest()[:16].upper()
                for _id in generated_ids
            ]

        embeddings = self._embed_documents(texts)
        if not metadatas:
            metadatas = [{} for _ in texts]
        docs = [
            (id_, text, json.dumps(metadata), array.array("f", embedding))
            for id_, text, metadata, embedding in zip(
                processed_ids, texts, metadatas, embeddings
            )
        ]

        with self.client.cursor() as cursor:
            cursor.executemany(
                f"INSERT INTO {self.table_name} (id, text, metadata, "
                f"embedding) VALUES (:1, :2, :3, :4)",
                docs,
            )
            self.client.commit()
        return processed_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        if isinstance(self.embedding_function, Embeddings):
            embedding = self.embedding_function.embed_query(query)
        documents = self.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return documents

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        if isinstance(self.embedding_function, Embeddings):
            embedding = self.embedding_function.embed_query(query)
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs_and_scores

    @_handle_exceptions
    def _get_clob_value(self, result: Any) -> str:
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        clob_value = ""
        if result:
            if isinstance(result, oracledb.LOB):
                raw_data = result.read()
                if isinstance(raw_data, bytes):
                    clob_value = raw_data.decode(
                        "utf-8"
                    )  # Specify the correct encoding
                else:
                    clob_value = raw_data
            elif isinstance(result, str):
                clob_value = result
            else:
                raise Exception("Unexpected type:", type(result))
        return clob_value

    @_handle_exceptions
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = []
        embedding_arr = array.array("f", embedding)

        query = f"""
        SELECT id,
          text,
          metadata,
          vector_distance(embedding, :embedding,
          {_get_distance_function(self.distance_strategy)}) as distance
        FROM {self.table_name}
        ORDER BY distance
        FETCH APPROX FIRST :k ROWS ONLY
        """
        # Execute the query
        with self.client.cursor() as cursor:
            cursor.execute(query, embedding=embedding_arr, k=k)
            results = cursor.fetchall()

            # Filter results if filter is provided
            for result in results:
                metadata = json.loads(
                    self._get_clob_value(result[2]) if result[2] is not None else "{}"
                )

                # Apply filtering based on the 'filter' dictionary
                if filter:
                    if all(metadata.get(key) in value for key, value in filter.items()):
                        doc = Document(
                            page_content=(
                                self._get_clob_value(result[1])
                                if result[1] is not None
                                else ""
                            ),
                            metadata=metadata,
                        )
                        distance = result[3]
                        docs_and_scores.append((doc, distance))
                else:
                    doc = Document(
                        page_content=(
                            self._get_clob_value(result[1])
                            if result[1] is not None
                            else ""
                        ),
                        metadata=metadata,
                    )
                    distance = result[3]
                    docs_and_scores.append((doc, distance))

        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, np.ndarray[np.float32, Any]]]:
        documents = []
        embedding_arr = array.array("f", embedding)

        query = f"""
        SELECT id,
          text,
          metadata,
          vector_distance(embedding, :embedding, {_get_distance_function(
            self.distance_strategy)}) as distance,
          embedding
        FROM {self.table_name}
        ORDER BY distance
        FETCH APPROX FIRST :k ROWS ONLY
        """

        # Execute the query
        with self.client.cursor() as cursor:
            cursor.execute(query, embedding=embedding_arr, k=k)
            results = cursor.fetchall()

            for result in results:
                page_content_str = self._get_clob_value(result[1])
                metadata_str = self._get_clob_value(result[2])
                metadata = json.loads(metadata_str)

                # Apply filter if provided and matches; otherwise, add all
                # documents
                if not filter or all(
                    metadata.get(key) in value for key, value in filter.items()
                ):
                    document = Document(
                        page_content=page_content_str, metadata=metadata
                    )
                    distance = result[3]
                    # Assuming result[4] is already in the correct format;
                    # adjust if necessary
                    current_embedding = (
                        np.array(result[4], dtype=np.float32)
                        if result[4]
                        else np.empty(0, dtype=np.float32)
                    )
                    documents.append((document, distance, current_embedding))
        return documents  # type: ignore

    @_handle_exceptions
    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the
        maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch before filtering to
                   pass to MMR algorithm.
          filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults
          to None.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal
            marginal
            relevance and score for each.
        """

        # Fetch documents and their scores
        docs_scores_embeddings = self.similarity_search_by_vector_returning_embeddings(
            embedding, fetch_k, filter=filter
        )
        # Assuming documents_with_scores is a list of tuples (Document, score)

        # If you need to split documents and scores for processing (e.g.,
        # for MMR calculation)
        documents, scores, embeddings = (
            zip(*docs_scores_embeddings) if docs_scores_embeddings else ([], [], [])
        )

        # Assume maximal_marginal_relevance method accepts embeddings and
        # scores, and returns indices of selected docs
        mmr_selected_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            list(embeddings),
            k=k,
            lambda_mult=lambda_mult,
        )

        # Filter documents based on MMR-selected indices and map scores
        mmr_selected_documents_with_scores = [
            (documents[i], scores[i]) for i in mmr_selected_indices
        ]

        return mmr_selected_documents_with_scores

    @_handle_exceptions
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: Optional[Dict[str, Any]]
          **kwargs: Any
        Returns:
          List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    @_handle_exceptions
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          query: Text to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: Optional[Dict[str, Any]]
          **kwargs
        Returns:
          List of Documents selected by maximal marginal relevance.

        `max_marginal_relevance_search` requires that `query` returns matched
        embeddings alongside the match documents.
        """
        embedding = self._embed_query(query)
        documents = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return documents

    @_handle_exceptions
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.
        Args:
          self: An instance of the class
          ids: List of ids to delete.
          **kwargs
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        # Compute SHA-256 hashes of the ids and truncate them
        hashed_ids = [
            hashlib.sha256(_id.encode()).hexdigest()[:16].upper() for _id in ids
        ]

        # Constructing the SQL statement with individual placeholders
        placeholders = ", ".join([":id" + str(i + 1) for i in range(len(hashed_ids))])

        ddl = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"

        # Preparing bind variables
        bind_vars = {
            f"id{i}": hashed_id for i, hashed_id in enumerate(hashed_ids, start=1)
        }

        with self.client.cursor() as cursor:
            cursor.execute(ddl, bind_vars)
            self.client.commit()

    @classmethod
    @_handle_exceptions
    def from_texts(
        cls: Type[OracleVS],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> OracleVS:
        """Return VectorStore initialized from texts and embeddings."""
        client = kwargs.get("client")
        if client is None:
            raise ValueError("client parameter is required...")
        params = kwargs.get("params", {})

        table_name = str(kwargs.get("table_name", "langchain"))

        distance_strategy = cast(
            DistanceStrategy, kwargs.get("distance_strategy", None)
        )
        if not isinstance(distance_strategy, DistanceStrategy):
            raise TypeError(
                f"Expected DistanceStrategy got " f"{type(distance_strategy).__name__} "
            )

        query = kwargs.get("query", "What is a Oracle database")

        drop_table_purge(client, table_name)

        vss = cls(
            client=client,
            embedding_function=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            query=query,
            params=params,
        )
        vss.add_texts(texts=list(texts), metadatas=metadatas)
        return vss
