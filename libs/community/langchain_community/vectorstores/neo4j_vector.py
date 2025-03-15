from __future__ import annotations

import enum
import logging
import os
from hashlib import md5
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_MAPPING = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "euclidean",
    DistanceStrategy.COSINE: "cosine",
}

COMPARISONS_TO_NATIVE = {
    "$eq": "=",
    "$ne": "<>",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

SPECIAL_CASED_OPERATORS = {
    "$in",
    "$nin",
    "$between",
}

TEXT_OPERATORS = {
    "$like",
    "$ilike",
}

LOGICAL_OPERATORS = {"$and", "$or"}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(TEXT_OPERATORS)
    .union(LOGICAL_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
)


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.SearchType",
)
class SearchType(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    VECTOR = "vector"
    HYBRID = "hybrid"


DEFAULT_SEARCH_TYPE = SearchType.VECTOR


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.IndexType",
)
class IndexType(str, enum.Enum):
    """Enumerator of the index types."""

    NODE = "NODE"
    RELATIONSHIP = "RELATIONSHIP"


DEFAULT_INDEX_TYPE = IndexType.NODE


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector._get_search_index_query",
)
def _get_search_index_query(
    search_type: SearchType, index_type: IndexType = DEFAULT_INDEX_TYPE
) -> str:
    if index_type == IndexType.NODE:
        type_to_query_map = {
            SearchType.VECTOR: (
                "CALL db.index.vector.queryNodes($index, $k, $embedding) "
                "YIELD node, score "
            ),
            SearchType.HYBRID: (
                "CALL { "
                "CALL db.index.vector.queryNodes($index, $k, $embedding) "
                "YIELD node, score "
                "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
                "UNWIND nodes AS n "
                # We use 0 as min
                "RETURN n.node AS node, (n.score / max) AS score UNION "
                "CALL db.index.fulltext.queryNodes($keyword_index, $query, "
                "{limit: $k}) YIELD node, score "
                "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
                "UNWIND nodes AS n "
                # We use 0 as min
                "RETURN n.node AS node, (n.score / max) AS score "
                "} "
                # dedup
                "WITH node, max(score) AS score ORDER BY score DESC LIMIT $k "
            ),
        }
        return type_to_query_map[search_type]
    else:
        return (
            "CALL db.index.vector.queryRelationships($index, $k, $embedding) "
            "YIELD relationship, score "
        )


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.check_if_not_null",
)
def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if the values are not None or empty string"""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.sort_by_index_name",
)
def sort_by_index_name(
    lst: List[Dict[str, Any]], index_name: str
) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists"""
    return sorted(lst, key=lambda x: x.get("name") != index_name)


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.remove_lucene_chars",
)
def remove_lucene_chars(text: str) -> str:
    """Remove Lucene special characters"""
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.dict_to_yaml_str",
)
def dict_to_yaml_str(input_dict: Dict, indent: int = 0) -> str:
    """
    Convert a dictionary to a YAML-like string without using external libraries.

    Parameters:
    - input_dict (dict): The dictionary to convert.
    - indent (int): The current indentation level.

    Returns:
    - str: The YAML-like string representation of the input dictionary.
    """
    yaml_str = ""
    for key, value in input_dict.items():
        padding = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{padding}{key}:\n{dict_to_yaml_str(value, indent + 1)}"
        elif isinstance(value, list):
            yaml_str += f"{padding}{key}:\n"
            for item in value:
                yaml_str += f"{padding}- {item}\n"
        else:
            yaml_str += f"{padding}{key}: {value}\n"
    return yaml_str


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.combine_queries",
)
def combine_queries(
    input_queries: List[Tuple[str, Dict[str, Any]]], operator: str
) -> Tuple[str, Dict[str, Any]]:
    """Combine multiple queries with an operator."""

    # Initialize variables to hold the combined query and parameters
    combined_query: str = ""
    combined_params: Dict = {}
    param_counter: Dict = {}

    for query, params in input_queries:
        # Process each query fragment and its parameters
        new_query = query
        for param, value in params.items():
            # Update the parameter name to ensure uniqueness
            if param in param_counter:
                param_counter[param] += 1
            else:
                param_counter[param] = 1
            new_param_name = f"{param}_{param_counter[param]}"

            # Replace the parameter in the query fragment
            new_query = new_query.replace(f"${param}", f"${new_param_name}")
            # Add the parameter to the combined parameters dictionary
            combined_params[new_param_name] = value

        # Combine the query fragments with an AND operator
        if combined_query:
            combined_query += f" {operator} "
        combined_query += f"({new_query})"

    return combined_query, combined_params


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.collect_params",
)
def collect_params(
    input_data: List[Tuple[str, Dict[str, str]]],
) -> Tuple[List[str], Dict[str, Any]]:
    """Transform the input data into the desired format.

    Args:
    - input_data (list of tuples): Input data to transform.
      Each tuple contains a string and a dictionary.

    Returns:
    - tuple: A tuple containing a list of strings and a dictionary.
    """
    # Initialize variables to hold the output parts
    query_parts = []
    params = {}

    # Loop through each item in the input data
    for query_part, param in input_data:
        # Append the query part to the list
        query_parts.append(query_part)
        # Update the params dictionary with the param dictionary
        params.update(param)

    # Return the transformed data
    return (query_parts, params)


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector._handle_field_filter",
)
def _handle_field_filter(
    field: str, value: Any, param_number: int = 1
) -> Tuple[str, Dict]:
    """Create a filter for a specific field.

    Args:
        field: name of field
        value: value to filter
            If provided as is then this will be an equality filter
            If provided as a dictionary then this will be a filter, the key
            will be the operator and the value will be the value to filter by
        param_number: sequence number of parameters used to map between param
           dict and Cypher snippet

    Returns a tuple of
        - Cypher filter snippet
        - Dictionary with parameters used in filter snippet
    """
    if not isinstance(field, str):
        raise ValueError(
            f"field should be a string but got: {type(field)} with value: {field}"
        )

    if field.startswith("$"):
        raise ValueError(
            f"Invalid filter condition. Expected a field but got an operator: {field}"
        )

    # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
    if not field.isidentifier():
        raise ValueError(f"Invalid field name: {field}. Expected a valid identifier.")

    if isinstance(value, dict):
        # This is a filter specification
        if len(value) != 1:
            raise ValueError(
                "Invalid filter condition. Expected a value which "
                "is a dictionary with a single key that corresponds to an operator "
                f"but got a dictionary with {len(value)} keys. The first few "
                f"keys are: {list(value.keys())[:3]}"
            )
        operator, filter_value = list(value.items())[0]
        # Verify that that operator is an operator
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Invalid operator: {operator}. Expected one of {SUPPORTED_OPERATORS}"
            )
    else:  # Then we assume an equality operator
        operator = "$eq"
        filter_value = value

    if operator in COMPARISONS_TO_NATIVE:
        # Then we implement an equality filter
        # native is trusted input
        native = COMPARISONS_TO_NATIVE[operator]
        query_snippet = f"n.`{field}` {native} $param_{param_number}"
        query_param = {f"param_{param_number}": filter_value}
        return (query_snippet, query_param)
    elif operator == "$between":
        low, high = filter_value
        query_snippet = (
            f"$param_{param_number}_low <= n.`{field}` <= $param_{param_number}_high"
        )
        query_param = {
            f"param_{param_number}_low": low,
            f"param_{param_number}_high": high,
        }
        return (query_snippet, query_param)

    elif operator in {"$in", "$nin", "$like", "$ilike"}:
        # We'll do force coercion to text
        if operator in {"$in", "$nin"}:
            for val in filter_value:
                if not isinstance(val, (str, int, float)):
                    raise NotImplementedError(
                        f"Unsupported type: {type(val)} for value: {val}"
                    )
        if operator in {"$in"}:
            query_snippet = f"n.`{field}` IN $param_{param_number}"
            query_param = {f"param_{param_number}": filter_value}
            return (query_snippet, query_param)
        elif operator in {"$nin"}:
            query_snippet = f"n.`{field}` NOT IN $param_{param_number}"
            query_param = {f"param_{param_number}": filter_value}
            return (query_snippet, query_param)
        elif operator in {"$like"}:
            query_snippet = f"n.`{field}` CONTAINS $param_{param_number}"
            query_param = {f"param_{param_number}": filter_value.rstrip("%")}
            return (query_snippet, query_param)
        elif operator in {"$ilike"}:
            query_snippet = f"toLower(n.`{field}`) CONTAINS $param_{param_number}"
            query_param = {f"param_{param_number}": filter_value.rstrip("%")}
            return (query_snippet, query_param)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.vectorstores.neo4j_vector.construct_metadata_filter",
)
def construct_metadata_filter(filter: Dict[str, Any]) -> Tuple[str, Dict]:
    """Construct a metadata filter.

    Args:
        filter: A dictionary representing the filter condition.

    Returns:
        Tuple[str, Dict]
    """

    if isinstance(filter, dict):
        if len(filter) == 1:
            # The only operators allowed at the top level are $AND and $OR
            # First check if an operator or a field
            key, value = list(filter.items())[0]
            if key.startswith("$"):
                # Then it's an operator
                if key.lower() not in ["$and", "$or"]:
                    raise ValueError(
                        f"Invalid filter condition. Expected $and or $or but got: {key}"
                    )
            else:
                # Then it's a field
                return _handle_field_filter(key, filter[key])

            # Here we handle the $and and $or operators
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected a list, but got {type(value)} for value: {value}"
                )
            if key.lower() == "$and":
                and_ = combine_queries(
                    [construct_metadata_filter(el) for el in value], "AND"
                )
                if len(and_) >= 1:
                    return and_
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            elif key.lower() == "$or":
                or_ = combine_queries(
                    [construct_metadata_filter(el) for el in value], "OR"
                )
                if len(or_) >= 1:
                    return or_
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            else:
                raise ValueError(
                    f"Invalid filter condition. Expected $and or $or but got: {key}"
                )
        elif len(filter) > 1:
            # Then all keys have to be fields (they cannot be operators)
            for key in filter.keys():
                if key.startswith("$"):
                    raise ValueError(
                        f"Invalid filter condition. Expected a field but got: {key}"
                    )
            # These should all be fields and combined using an $and operator
            and_multiple = collect_params(
                [
                    _handle_field_filter(k, v, index)
                    for index, (k, v) in enumerate(filter.items())
                ]
            )
            if len(and_multiple) >= 1:
                return " AND ".join(and_multiple[0]), and_multiple[1]
            else:
                raise ValueError(
                    "Invalid filter condition. Expected a dictionary "
                    "but got an empty dictionary"
                )
        else:
            raise ValueError("Got an empty dictionary for filters.")


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.Neo4jVector",
)
class Neo4jVector(VectorStore):
    """`Neo4j` vector index.

    To use, you should have the ``neo4j`` python package installed.

    Args:
        url: Neo4j connection url
        username: Neo4j username.
        password: Neo4j password
        database: Optionally provide Neo4j database
                  Defaults to "neo4j"
        embedding: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        distance_strategy: The distance strategy to use. (default: COSINE)
        search_type: The type of search to be performed, either
            'vector' or 'hybrid'
        node_label: The label used for nodes in the Neo4j database.
            (default: "Chunk")
        embedding_node_property: The property name in Neo4j to store embeddings.
            (default: "embedding")
        text_node_property: The property name in Neo4j to store the text.
            (default: "text")
        retrieval_query: The Cypher query to be used for customizing retrieval.
            If empty, a default query will be used.
        index_type: The type of index to be used, either
            'NODE' or 'RELATIONSHIP'
        pre_delete_collection: If True, will delete existing data if it exists.
            (default: False). Useful for testing.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores.neo4j_vector import Neo4jVector
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            url="bolt://localhost:7687"
            username="neo4j"
            password="pleaseletmein"
            embeddings = OpenAIEmbeddings()
            vectorestore = Neo4jVector.from_documents(
                embedding=embeddings,
                documents=docs,
                url=url
                username=username,
                password=password,
            )


    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        search_type: SearchType = SearchType.VECTOR,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        keyword_index_name: Optional[str] = "keyword",
        database: Optional[str] = None,
        index_name: str = "vector",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        logger: Optional[logging.Logger] = None,
        pre_delete_collection: bool = False,
        retrieval_query: str = "",
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        index_type: IndexType = DEFAULT_INDEX_TYPE,
        graph: Optional[Neo4jGraph] = None,
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        # Allow only cosine and euclidean distance strategies
        if distance_strategy not in [
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.COSINE,
        ]:
            raise ValueError(
                "distance_strategy must be either 'EUCLIDEAN_DISTANCE' or 'COSINE'"
            )

        # Graph object takes precedent over env or input params
        if graph:
            self._driver = graph._driver
            self._database = graph._database
        else:
            # Handle if the credentials are environment variables
            # Support URL for backwards compatibility
            if not url:
                url = os.environ.get("NEO4J_URL")

            url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
            username = get_from_dict_or_env(
                {"username": username}, "username", "NEO4J_USERNAME"
            )
            password = get_from_dict_or_env(
                {"password": password}, "password", "NEO4J_PASSWORD"
            )
            database = get_from_dict_or_env(
                {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
            )

            self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
            self._database = database
            # Verify connection
            try:
                self._driver.verify_connectivity()
            except neo4j.exceptions.ServiceUnavailable:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the url is correct"
                )
            except neo4j.exceptions.AuthError:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the username and password are correct"
                )

        self.schema = ""
        # Verify if the version support vector index
        self._is_enterprise = False
        self.verify_version()

        # Verify that required values are not null
        check_if_not_null(
            [
                "index_name",
                "node_label",
                "embedding_node_property",
                "text_node_property",
            ],
            [index_name, node_label, embedding_node_property, text_node_property],
        )

        self.embedding = embedding
        self._distance_strategy = distance_strategy
        self.index_name = index_name
        self.keyword_index_name = keyword_index_name
        self.node_label = node_label
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.retrieval_query = retrieval_query
        self.search_type = search_type
        self._index_type = index_type
        # Calculate embedding dimension
        self.embedding_dimension = len(embedding.embed_query("foo"))

        # Delete existing data if flagged
        if pre_delete_collection:
            from neo4j.exceptions import DatabaseError

            self.query(
                f"MATCH (n:`{self.node_label}`) "
                "CALL (n) { DETACH DELETE n } "
                "IN TRANSACTIONS OF 10000 ROWS;"
            )
            # Delete index
            try:
                self.query(f"DROP INDEX {self.index_name}")
            except DatabaseError:  # Index didn't exist yet
                pass

    def query(
        self,
        query: str,
        *,
        params: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Query Neo4j database with retries and exponential backoff.

        Args:
            query (str): The Cypher query to execute.
            params (dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results.
        """
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        params = params or {}
        try:
            data, _, _ = self._driver.execute_query(
                query, database_=self._database, parameters_=params
            )
            return [r.data() for r in data]
        except Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and "in an implicit transaction" in e.message  # type: ignore[operator]
                )
                or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and (
                        "in an open transaction is not possible" in e.message  # type: ignore[operator]
                        or "tried to execute in an explicit transaction" in e.message  # type: ignore[operator]
                    )
                )
            ):
                raise
        # Fallback to allow implicit transactions
        with self._driver.session(database=self._database) as session:
            data = session.run(Query(text=query), params)  # type: ignore[assignment]
            return [r.data() for r in data]

    def verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        db_data = self.query("CALL dbms.components()")
        version = db_data[0]["versions"][0]
        if "aura" in version:
            version_tuple = tuple(map(int, version.split("-")[0].split("."))) + (0,)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

        # Flag for metadata filtering
        metadata_target_version = (5, 18, 0)
        if version_tuple < metadata_target_version:
            self.support_metadata_filter = False
        else:
            self.support_metadata_filter = True
        # Flag for enterprise
        self._is_enterprise = True if db_data[0]["edition"] == "enterprise" else False

    def retrieve_existing_index(self) -> Tuple[Optional[int], Optional[str]]:
        """
        Check if the vector index exists in the Neo4j database
        and returns its embedding dimension.

        This method queries the Neo4j database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """

        index_information = self.query(
            "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, "
            "properties, options WHERE type = 'VECTOR' AND (name = $index_name "
            "OR (labelsOrTypes[0] = $node_label AND "
            "properties[0] = $embedding_node_property)) "
            "RETURN name, entityType, labelsOrTypes, properties, options ",
            params={
                "index_name": self.index_name,
                "node_label": self.node_label,
                "embedding_node_property": self.embedding_node_property,
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.index_name = index_information[0]["name"]
            self.node_label = index_information[0]["labelsOrTypes"][0]
            self.embedding_node_property = index_information[0]["properties"][0]
            self._index_type = index_information[0]["entityType"]
            embedding_dimension = None
            index_config = index_information[0]["options"]["indexConfig"]
            if "vector.dimensions" in index_config:
                embedding_dimension = index_config["vector.dimensions"]

            return embedding_dimension, index_information[0]["entityType"]
        except IndexError:
            return None, None

    def retrieve_existing_fts_index(
        self, text_node_properties: List[str] = []
    ) -> Optional[str]:
        """
        Check if the fulltext index exists in the Neo4j database

        This method queries the Neo4j database for existing fts indexes
        with the specified name.

        Returns:
            (Tuple): keyword index information
        """

        index_information = self.query(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
            "WHERE type = 'FULLTEXT' AND (name = $keyword_index_name "
            "OR (labelsOrTypes = [$node_label] AND "
            "properties = $text_node_property)) "
            "RETURN name, labelsOrTypes, properties, options ",
            params={
                "keyword_index_name": self.keyword_index_name,
                "node_label": self.node_label,
                "text_node_property": text_node_properties or [self.text_node_property],
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.keyword_index_name = index_information[0]["name"]
            self.text_node_property = index_information[0]["properties"][0]
            node_label = index_information[0]["labelsOrTypes"][0]
            return node_label
        except IndexError:
            return None

    def create_new_index(self) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in Neo4j.
        """
        index_query = (
            f"CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS "
            f"FOR (m:`{self.node_label}`) ON m.`{self.embedding_node_property}` "
            "OPTIONS { indexConfig: { "
            "`vector.dimensions`: toInteger($embedding_dimension), "
            "`vector.similarity_function`: $similarity_metric }}"
        )

        parameters = {
            "embedding_dimension": self.embedding_dimension,
            "similarity_metric": DISTANCE_MAPPING[self._distance_strategy],
        }
        self.query(index_query, params=parameters)

    def create_new_keyword_index(self, text_node_properties: List[str] = []) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new full text index in Neo4j.
        """
        node_props = text_node_properties or [self.text_node_property]
        fts_index_query = (
            f"CREATE FULLTEXT INDEX {self.keyword_index_name} "
            f"FOR (n:`{self.node_label}`) ON EACH "
            f"[{', '.join(['n.`' + el + '`' for el in node_props])}]"
        )
        self.query(fts_index_query)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        create_id_index: bool = True,
        search_type: SearchType = SearchType.VECTOR,
        **kwargs: Any,
    ) -> Neo4jVector:
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            embedding=embedding,
            search_type=search_type,
            **kwargs,
        )
        # Check if the vector index already exists
        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Data ingestion is not supported with relationship vector index."
            )

        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index with name {store.index_name} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index()
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Create unique constraint for faster import
        if create_id_index:
            store.query(
                "CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:`{store.node_label}`) REQUIRE n.id IS UNIQUE;"
            )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        import_query = (
            "UNWIND $data AS row "
            "CALL (row) { WITH row "
            f"MERGE (c:`{self.node_label}` {{id: row.id}}) "
            "WITH c, row "
            f"CALL db.create.setNodeVectorProperty(c, "
            f"'{self.embedding_node_property}', row.embedding) "
            f"SET c.`{self.text_node_property}` = row.text "
            "SET c += row.metadata "
            "} IN TRANSACTIONS OF 1000 ROWS "
        )

        parameters = {
            "data": [
                {"text": text, "metadata": metadata, "embedding": embedding, "id": id}
                for text, metadata, embedding, id in zip(
                    texts, metadatas, embeddings, ids
                )
            ]
        }

        self.query(import_query, params=parameters)

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        params: Dict[str, Any] = {},
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Neo4jVector.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        params: Dict[str, Any] = {},
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            **kwargs,
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the Neo4j database using a
        given vector and return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that
        are most similar to a given embedding. The similarity is measured
        using a vector index in the Neo4j database. The results are returned
        as a list of tuples, each containing a Document object and
        its similarity score.

        Args:
            embedding (List[float]): The embedding vector to compare against.
            k (int, optional): The number of top similar documents to retrieve.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                a Document object and its similarity score.
        """
        if filter:
            # Verify that 5.18 or later is used
            if not self.support_metadata_filter:
                raise ValueError(
                    "Metadata filtering is only supported in "
                    "Neo4j version 5.18 or greater"
                )
            # Metadata filtering and hybrid doesn't work
            if self.search_type == SearchType.HYBRID:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )
            parallel_query = (
                "CYPHER runtime = parallel parallelRuntimeSupport=all "
                if self._is_enterprise
                else ""
            )
            base_index_query = parallel_query + (
                f"MATCH (n:`{self.node_label}`) WHERE "
                f"n.`{self.embedding_node_property}` IS NOT NULL AND "
                f"size(n.`{self.embedding_node_property}`) = "
                f"toInteger({self.embedding_dimension}) AND "
            )
            base_cosine_query = (
                " WITH n as node, vector.similarity.cosine("
                f"n.`{self.embedding_node_property}`, "
                "$embedding) AS score ORDER BY score DESC LIMIT toInteger($k) "
            )
            filter_snippets, filter_params = construct_metadata_filter(filter)
            index_query = base_index_query + filter_snippets + base_cosine_query

        else:
            index_query = _get_search_index_query(self.search_type, self._index_type)
            filter_params = {}

        if self._index_type == IndexType.RELATIONSHIP:
            if kwargs.get("return_embeddings"):
                default_retrieval = (
                    f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                    f"relationship {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null, "
                    f"_embedding_: relationship.`{self.embedding_node_property}`}} "
                    "AS metadata"
                )
            else:
                default_retrieval = (
                    f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                    f"relationship {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
                )

        else:
            if kwargs.get("return_embeddings"):
                default_retrieval = (
                    f"RETURN node.`{self.text_node_property}` AS text, score, "
                    f"node {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null, "
                    f"_embedding_: node.`{self.embedding_node_property}`}} AS metadata"
                )
            else:
                default_retrieval = (
                    f"RETURN node.`{self.text_node_property}` AS text, score, "
                    f"node {{.*, `{self.text_node_property}`: Null, "
                    f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
                )

        retrieval_query = (
            self.retrieval_query if self.retrieval_query else default_retrieval
        )

        read_query = index_query + retrieval_query
        parameters = {
            "index": self.index_name,
            "k": k,
            "embedding": embedding,
            "keyword_index": self.keyword_index_name,
            "query": remove_lucene_chars(kwargs["query"]),
            **params,
            **filter_params,
        }

        results = self.query(read_query, params=parameters)

        if any(result["text"] is None for result in results):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.text_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `text` column"
                )
        if kwargs.get("return_embeddings") and any(
            result["metadata"]["_embedding_"] is None for result in results
        ):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.embedding_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `_embedding_` metadata column"
                )

        docs = [
            (
                Document(
                    page_content=dict_to_yaml_str(result["text"])
                    if isinstance(result["text"], dict)
                    else result["text"],
                    metadata={
                        k: v for k, v in result["metadata"].items() if v is not None
                    },
                ),
                result["score"],
            )
            for result in results
        ]
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, params=params, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[Neo4jVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Return Neo4jVector initialized from texts and embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Neo4jVector:
        """Construct Neo4jVector wrapper from raw documents and pre-
        generated embeddings.

        Return Neo4jVector initialized from documents and embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores.neo4j_vector import Neo4jVector
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                vectorstore = Neo4jVector.from_embeddings(
                    text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        index_name: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        keyword_index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Get instance of an existing Neo4j vector index. This method will
        return the instance of the store without inserting any new
        embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters along with
        the `index_name` definition.
        """

        if search_type == SearchType.HYBRID and not keyword_index_name:
            raise ValueError(
                "keyword_index name has to be specified when using hybrid search option"
            )

        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            **kwargs,
        )

        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Relationship vector index is not supported with "
                "`from_existing_index` method. Please use the "
                "`from_existing_relationship_index` method."
            )

        if not index_type:
            raise ValueError(
                "The specified vector index name does not exist. "
                "Make sure to check if you spelled it correctly"
            )

        # Check if embedding function and vector index dimensions match
        if embedding_dimension and not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                raise ValueError(
                    "The specified keyword index name does not exist. "
                    "Make sure to check if you spelled it correctly"
                )
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        return store

    @classmethod
    def from_existing_relationship_index(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        index_name: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Get instance of an existing Neo4j relationship vector index.
        This method will return the instance of the store without
        inserting any new embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters along with
        the `index_name` definition.
        """

        if search_type == SearchType.HYBRID:
            raise ValueError(
                "Hybrid search is not supported in combination "
                "with relationship vector index"
            )

        store = cls(
            embedding=embedding,
            index_name=index_name,
            **kwargs,
        )

        embedding_dimension, index_type = store.retrieve_existing_index()

        if not index_type:
            raise ValueError(
                "The specified vector index name does not exist. "
                "Make sure to check if you spelled it correctly"
            )
        # Raise error if relationship index type
        if index_type == "NODE":
            raise ValueError(
                "Node vector index is not supported with "
                "`from_existing_relationship_index` method. Please use the "
                "`from_existing_index` method."
            )

        # Check if embedding function and vector index dimensions match
        if embedding_dimension and not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        return store

    @classmethod
    def from_documents(
        cls: Type[Neo4jVector],
        documents: List[Document],
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Return Neo4jVector initialized from documents and embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_existing_graph(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        *,
        keyword_index_name: Optional[str] = "keyword",
        index_name: str = "vector",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Initialize and return a Neo4jVector instance from an existing graph.

        This method initializes a Neo4jVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
        Neo4jVector: An instance of Neo4jVector initialized with the provided parameters
                    and existing graph.

        Example:
        >>> neo4j_vector = Neo4jVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        Neo4j credentials are required in the form of `url`, `username`, and `password`,
        and optional `database` parameters passed as additional keyword arguments.
        """
        # Validate the list is not empty
        if not text_node_properties:
            raise ValueError(
                "Parameter `text_node_properties` must not be an empty list"
            )
        # Prefer retrieval query from params, otherwise construct it
        if not retrieval_query:
            retrieval_query = (
                f"RETURN reduce(str='', k IN {text_node_properties} |"
                " str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, "
                "node {.*, `"
                + embedding_node_property
                + "`: Null, id: Null, "
                + ", ".join([f"`{prop}`: Null" for prop in text_node_properties])
                + "} AS metadata, score"
            )
        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            **kwargs,
        )

        # Check if the vector index already exists
        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "`from_existing_graph` method does not support "
                " existing relationship vector index. "
                "Please use `from_existing_relationship_index` method"
            )

        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index with name {store.index_name} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        # FTS index for Hybrid search
        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index(text_node_properties)
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index(text_node_properties)
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Populate embeddings
        while True:
            fetch_query = (
                f"MATCH (n:`{node_label}`) "
                f"WHERE n.{embedding_node_property} IS null "
                "AND any(k in $props WHERE n[k] IS NOT null) "
                f"RETURN elementId(n) AS id, reduce(str='',"
                "k IN $props | str + '\\n' + k + ':' + coalesce(n[k], '')) AS text "
                "LIMIT 1000"
            )
            data = store.query(fetch_query, params={"props": text_node_properties})
            if not data:
                break
            text_embeddings = embedding.embed_documents([el["text"] for el in data])

            params = {
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(data, text_embeddings)
                ]
            }

            store.query(
                "UNWIND $data AS row "
                f"MATCH (n:`{node_label}`) "
                "WHERE elementId(n) = row.id "
                f"CALL db.create.setNodeVectorProperty(n, "
                f"'{embedding_node_property}', row.embedding) "
                "RETURN count(*)",
                params=params,
            )
            # If embedding calculation should be stopped
            if len(data) < 1000:
                break
        return store

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        # Embed the query
        query_embedding = self.embedding.embed_query(query)

        # Fetch the initial documents
        got_docs = self.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            query=query,
            k=fetch_k,
            return_embeddings=True,
            filter=filter,
            **kwargs,
        )

        # Get the embeddings for the fetched documents
        got_embeddings = [doc.metadata["_embedding_"] for doc, _ in got_docs]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), got_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_docs = [got_docs[i][0] for i in selected_indices]

        # Remove embedding values from metadata
        for doc in selected_docs:
            del doc.metadata["_embedding_"]

        return selected_docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return lambda x: x
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return lambda x: x
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )
