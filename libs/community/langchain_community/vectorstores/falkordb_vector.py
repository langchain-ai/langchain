from __future__ import annotations
import random

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
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

from langchain_community.graphs import FalkorDBGraph
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
import redis
import redis.exceptions
import string

def generate_random_string(length: int) -> str:
    # Define the characters to use: uppercase, lowercase, digits, and punctuation
    characters = string.ascii_letters 
    # Randomly choose 'length' characters from the pool of possible characters
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string




DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN_DISTANCE
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


class SearchType(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    VECTOR = "vector"
    HYBRID = "hybrid"


DEFAULT_SEARCH_TYPE = SearchType.VECTOR

class IndexType(str, enum.Enum):
    """Enumerator of the index types."""

    NODE = "NODE"
    RELATIONSHIP = "RELATIONSHIP"


DEFAULT_INDEX_TYPE = IndexType.NODE


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


def _get_search_index_query(
        search_type: SearchType, index_type: IndexType = DEFAULT_INDEX_TYPE
) -> str:
    if index_type == IndexType.NODE:
        type_to_query_map = {
        SearchType.VECTOR: (
            "CALL db.idx.vector.queryNodes($entity_label, $entity_property, $k, vecf32($embedding)) "
            "YIELD node, score "
        ),
        SearchType.HYBRID: (
            "CALL { "
            "CALL db.idx.vector.queryNodes($entity_label, $entity_property, $k, vecf32($embedding)) "
            "YIELD node, score "
            "WITH collect({node: node, score: score}) AS nodes, max(score) AS max_score "
            "UNWIND nodes AS n "
            "RETURN n.node AS node, (n.score / max_score) AS score "
            "UNION "
            "CALL db.idx.fulltext.queryNodes($entity_label, $query) "
            "YIELD node, score "
            "WITH collect({node: node, score: score}) AS nodes, max(score) AS max_score "
            "UNWIND nodes AS n "
            "RETURN n.node AS node, (n.score / max_score) AS score "
            "} "
            "WITH node, max(score) AS score "
            "ORDER BY score DESC LIMIT $k "
        ),
    }
        return type_to_query_map[search_type]
    else:
        return (
            "CALL db.idx.vector.queryRelationships($entity_label, $entity_property, $k, vecf32($embedding)) "
            "YIELD relationship, score "
        )

def process_data(data: List[List[Any]]) -> List[Dict[str, Any]]:
    """
    Processes a nested list of entity data to extract information about  labels, entity types, properties, index types, 
    and index details (if applicable).

    Args:
        data (List[List[Any]]): A nested list containing details about entitys, their properties, 
                                index types, and configuration information.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary contains:
            - entity_label (str): The label or name of the entity or relationship (e.g., 'Person', 'Song').
            - entity_property (str): The property of the entity or relationship on which an index was created (e.g., 'first_name').
            - index_type (str or List[str]): The type(s) of index applied to the property (e.g., 'FULLTEXT', 'VECTOR').
            - index_status (str): The status of the index (e.g., 'OPERATIONAL', 'PENDING').
            - index_dimension (Optional[int]): The dimension of the vector index, if applicable.
            - index_similarityFunction (Optional[str]): The similarity function used by the vector index, if applicable.
            - entity_type (str): The type of entity. That is either entity or relationship
    
    Notes:
        - The entity label is extracted from the first element of each entity list.
        - The entity property and associated index types are extracted from the second element.
        - If the index type includes 'VECTOR', additional details such as dimension and similarity function 
          are extracted from the entity configuration.
        - The function handles cases where entitys have multiple index types (e.g., both 'FULLTEXT' and 'VECTOR').
    """

    result = []
    
    
    for entity in data:
        # Extract basic information
        
        entity_label = entity[0]

       
        entity_properties = entity[1]

        
        index_type_dict = entity[2]

        

        
        index_status = entity[7]

        
        entity_type = entity[6]
        
        # Process each property and its index type(s)
        for prop, index_types in index_type_dict.items():
            entity_info = {
                "entity_label": entity_label,
                "entity_property": prop,
                "entity_type": entity_type,
                "index_type": index_types[0],
                "index_status": index_status,
                "index_dimension": None,
                "index_similarityFunction": None
            }
            
            # Check for VECTOR type and extract additional details
            if "VECTOR" in index_types:
                if isinstance(entity[3], str):
                    entity_info["index_dimension"] = None
                    entity_info["index_similarityFunction"] = None
                else:
                    vector_info = entity[3].get(prop, {})
                    entity_info["index_dimension"] = vector_info.get('dimension')
                    entity_info["index_similarityFunction"] = vector_info.get('similarityFunction')

            result.append(entity_info)
    
    return result


def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if the values are not None or empty string"""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")







class FalkorDBVector(VectorStore):
    """`FalkorDB` vector index.

    To use, you should have the ``falkordb`` python package installed

    Args:
        host: FalkorDB host
        port: FalkorDB port
        username: Optionally provide your username 
                  details if you are connecting to a
                  FalkorDB Cloud database instance
        password: Optionally provide your password
                  details if you are connecting to a
                  FalkorDB Cloud database instance
        embedding: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
        distance_strategy The distance strategy to use. (default: "EUCLIDEAN")
        pre_delete_collection: If True, will delete existing data if it exists.
                (default: False). Useful for testing.


    Example:
        .. code-block:: python

            from langchain_community.vectorstores.falkordb_vector import FalkorDBVector
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            host="localhost"
            port=6379
            embeddings=OpenAIEmbeddings()
            vectorstore = FalkorDBVector.from_documents(
                embedding=embeddings,
                documents=docs,
                host=host,
                port=port,  
            )


        

    """
    def __init__(
        self,
        embedding: Embeddings,
        *,
        search_type: SearchType = SearchType.VECTOR,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        database: str = generate_random_string(4),
        node_label: str = "Chunk",
        relation_type: str = "",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        embedding_dimension: Optional[int] = None,
        retrieval_query: str = "",
        index_type: IndexType = DEFAULT_INDEX_TYPE,
        graph: Optional[FalkorDBGraph] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        ssl: bool  = False,
        pre_delete_collection: bool = False,
    ) -> None:
        try:
            import falkordb 
        except ImportError:
            raise ImportError(
                "Could not import falkordb python package."
                "Please install it with `pip install falkordb`"
            )
        
        # Only euclidean distance strategies is avaliable in falkor for now
        if distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE:
            raise ValueError(
                "distance_strategy must be 'EUCLIDEAN_DISTANCE'"
            )
        
        # Graph object takes precedent over env or input params
        if graph:
            self._database = graph._graph
            self._driver = graph._driver
        else:
            # Handle credentials via environment variables or input params
            self._host = host
            self._port = port
            self._username = username or os.environ.get("FALKORDB_USERNAME")
            self._password = password or os.environ.get("FALKORDB_PASSWORD")
            self._ssl = ssl

            # Initialize the FalkorDB connection
            # Initialize the FalkorDB connection
            try:
                self._driver = falkordb.FalkorDB(
                    host=self._host,
                    port=self._port,
                    username=self._username,
                    password=self._password,
                    ssl=self._ssl,
                )
            except redis.exceptions.ConnectionError:
                raise ValueError(
                    "Could not connect to FalkorDB database."
                    "Please ensure that the host and port is correct"
                )
            except redis.exceptions.AuthenticationError:
                raise ValueError(
                    "Could not connect to FalkorDB database. "
                    "Please ensure that the username and password are correct"
                )
            
            # Verify that required values are not null
            check_if_not_null(
                ["embedding_node_property","node_label"],
                [embedding_node_property, node_label],
            )

            print("Database name: ", database)
            self._database = self._driver.select_graph(database)
            self.database_name = database

            self.embedding = embedding
            self.node_label = node_label
            self.relation_type = relation_type
            self.embedding_node_property = embedding_node_property
            self.text_node_property = text_node_property
            self._distance_strategy = distance_strategy
            self.override_relevance_score_fn = relevance_score_fn
            #self.logger = logger
            self.pre_delete_collection = pre_delete_collection
            self.retrieval_query = retrieval_query
            #self.relevance_score_fn = relevance_score_fn
            self.search_type = search_type
            self._index_type = index_type
            self.metadata = []

            # Calculate embedding dimension
            if embedding_dimension == None:
                self.embedding_dimension = len(embedding.embed_query("foo"))
            else:
                self.embedding_dimension = embedding_dimension



            # Delete existing data if flagged
            if pre_delete_collection:
                self._database.query(f"""
MATCH (n:`{self.node_label}`)
DETACH DELETE n
""")


    def query(
            self,
            query: str,
            *,
            params: Optional[dict] = None,
            retry_on_timeout: bool = True,
    ) -> List[List]:
        """
        This method sends a Cypher query to the connected FalkorDB database
        and returns the results as a list of lists.

        Args:
            query (str): The Cypher query to execute.
            params (dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[List]: List of Lists containing the query results
        """
        params = params or {}
        try:
            data = self._database.query(query, params)
            return data.result_set
        except Exception as e:
            if "Invalid input" in str(e):
                raise ValueError(f"Cypher Statement is not valid\n{e}")
        except Exception as e:
            if retry_on_timeout:
                return self.query(
                    query, params=params, retry_on_timeout=False
                )
            else:
                raise e
            
    def create_new_index(
            self,
            node_label: Optional[str] = "",
            embedding_node_property: Optional[str] = "",
            embedding_dimension: Optional[int] = None
            ) -> None:
        """
        This method creates a new vector index
        on a node in FalkorDB.
        """
        if node_label:
            node_label = node_label
        elif self.node_label:
            node_label = self.node_label
        else:
            raise ValueError(
                "`node_label` property must be set to use this function"
            )
        
        if embedding_node_property:
            embedding_node_property = embedding_node_property
        elif self.embedding_node_property:
            embedding_node_property = self.embedding_node_property
        else:
            raise ValueError(
                "`embedding_node_property` property must be set to use this function"
            )
        
        if embedding_dimension:
            embedding_dimension = embedding_dimension
        elif self.embedding_dimension:
            embedding_dimension = self.embedding_dimension
        else:
            raise ValueError(
                "`node_label` property must be set to use this function"
            )
        try:
            self._database.create_node_vector_index(
                node_label,
                embedding_node_property,
                dim=embedding_dimension,
                similarity_function=DISTANCE_MAPPING[self._distance_strategy]
            )
            self._index_type = 'NODE'
        except Exception as e:
            if "already indexed" in str(e):
                print(
                    f"A vector index on (:{node_label}"
                    "{"
                    f"{embedding_node_property}"
                    "}"
                    ")"
                    " has already been created"
                      )

    def create_new_index_on_relationship(
            self,
            relation_type: str = "",
            embedding_node_property: str = "" ,
            embedding_dimension: int = None,
    ) -> None:
        """
        This method creates an new vector index
        on a relationship/edge in FalkorDB.
        """
        if not relation_type:
            raise ValueError(
                "`relation_type` must be set to use this function"
            )
        if not embedding_node_property:
            raise ValueError(
                "`embedding_node_property` must be set to use this function"
            )
        if not embedding_dimension:
            raise ValueError(
                "`embedding_dimension` must be set to use this function"
            )
        
        try:
            self._database.create_edge_vector_index(
                relation_type,
                embedding_node_property,
                dim=embedding_dimension,
                similarity_function=DISTANCE_MAPPING[DEFAULT_DISTANCE_STRATEGY]
            )
            self._index_type = 'RELATIONSHIP'
        except Exception as e:
            if "already indexed" in str(e):
                print(
                    f"A vector index on [:{relation_type}"
                    "{"
                    f"{embedding_node_property}"
                    "}"
                    "]"
                    " has already been created"
                      )

    def create_new_keyword_index(self, text_node_properties: List[str] = []) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new full text index in FalkorDB
        Args:
        text_node_properties (List[str]): List of node properties to be indexed. 
                                          If not provided, defaults to self.text_node_property.
        """
        # Use the provided properties or default to self.text_node_property
        node_props = text_node_properties or [self.text_node_property]
        
        # Dynamically pass node label and properties to create the full-text index
        self._database.create_node_fulltext_index(
            self.node_label,
            *node_props
        )


            
    def retrieve_existing_index(
            self,
            node_label: Optional[str] = ""
            ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """
        Check if the vector index exists in the FalkorDB database
        and returns its embedding dimension, entity_type, entity_label, entity_property

        This method;
        1. queries the FalkorDB database for existing indexes
        2. attempts to retrieve the dimension of the vector index with the specified node label & index type
        3. If the index exists, its dimension is returned.
        4. Else if the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found,
            str or None: The entity type found.
            str or None: The label of the entity that the vector index was created with
            str or None: The property of the entity for which the vector index was created on


        """
        if node_label:
            node_label = node_label
        elif self.node_label:
            node_label = self.node_label
        else:
            raise ValueError(
                "`node_label` property must be set to use this function"
            )
        
        embedding_dimension = None
        entity_type = None
        entity_label = None
        entity_property = None
        index_information = self._database.query("""
CALL db.indexes() 
""")
        if index_information:
            processed_index_information = process_data(index_information.result_set)
            for dict in processed_index_information:
                if dict.get('entity_label', False) == node_label:
                    if dict['index_type'] == 'VECTOR':
                        embedding_dimension = int(dict['index_dimension'])
                        entity_type = str(dict['entity_type'])
                        entity_label = str(dict['entity_label'])
                        entity_property = str(dict['entity_property'])
                        break
            if embedding_dimension and entity_type and entity_label and entity_property:
                self._index_type= entity_type
                return embedding_dimension, entity_type, entity_label, entity_property
            else:
                return None, None, None, None 
        else:
            return None, None, None, 

    def retrieve_existing_relationship_index(
            self,
            relation_type: Optional[str] = ""
            ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """
        Check if the vector index exists in the FalkorDB database
        and returns its embedding dimension, entity_type, entity_label, entity_property

        This method;
        1. queries the FalkorDB database for existing indexes
        2. attempts to retrieve the dimension of the vector index with the specified label & index type
        3. If the index exists, its dimension is returned.
        4. Else if the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found,
            str or None: The entity type found.
            str or None: The label of the entity that the vector index was created with
            str or None: The property of the entity for which the vector index was created on


        """

        if relation_type:
            relation_type = relation_type
        elif self.relation_type:
            relation_type = self.relation_type
        else:
            raise ValueError(
                "Couldn't find any specified `relation_type`. Check if you spelled it correctly"
            )
        
        embedding_dimension = None
        entity_type = None
        entity_label = None
        entity_property = None
        index_information = self._database.query("""
CALL db.indexes() 
""")
        if index_information:
            processed_index_information = process_data(index_information.result_set)
            for dict in processed_index_information:
                if dict.get('entity_label', False) == relation_type:
                    if dict['index_type'] == 'VECTOR':
                        embedding_dimension = int(dict['index_dimension'])
                        entity_type = str(dict['entity_type'])
                        entity_label = str(dict['entity_label'])
                        entity_property = str(dict['entity_property'])
                        break
            if embedding_dimension and entity_type and entity_label and entity_property:
                self._index_type= entity_type
                return embedding_dimension, entity_type, entity_label, entity_property
            else:
                return None, None, None, None 
        else:
            return None, None, None, None
        

    
    def retrieve_existing_fts_index(self) -> Optional[str]:
        """
        Check if the fulltext index exists in the FalkorDB database

        This method queries the FalkorDB database for existing fts indexes
        with the specified name.

        Returns:
            str: fulltext index entity label
        """

        entity_label = None
        index_information = self._database.query("""
CALL db.indexes() 
""")
        if index_information:
            processed_index_information = process_data(index_information.result_set)
            for dict in processed_index_information:
                if dict.get('entity_label', False) == self.node_label:
                    if dict['index_type'] == 'FULLTEXT':
                        entity_label = str(dict['entity_label'])
                        break

            if entity_label:
                return entity_label
            else:
                return None
        else:
            return None
    
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
        search_type: SearchType = SearchType.VECTOR,
        **kwargs: Any,
    ) -> FalkorDBVector:
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]


        dimension_for_me = {
            "data": [
                {"embedding": embedding} 
                for embedding in embeddings
            ]
        }

        store = cls(
            embedding=embedding,
            search_type=search_type,
            **kwargs,
        )

        # Access the first embedding store the length of the dimension
        store.embedding_dimension = len(dimension_for_me["data"][0]["embedding"])

        # Check if the vector index already exists
        embedding_dimension, index_type, entity_label, entity_property = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Data ingestion is not supported with relationship vector index"
            )
        
        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
            embedding_dimension, index_type, entity_label, entity_property = store.retrieve_existing_index()
            

        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"A Vector index for {entity_label} on {entity_property} exists"
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
            else: # Validate that FTS and Vector Index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
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


        self.metadata = []
        # Check if all dictionaries are empty
        if all(not metadata for metadata in metadatas):
            pass
        else:
            # Initialize a set to keep track of unique non-empty keys
            unique_non_empty_keys = set()

            # Iterate over each metadata dictionary
            for metadata in metadatas:
                # Add keys with non-empty values to the set
                unique_non_empty_keys.update(key for key, value in metadata.items() if value)
            
            # Print unique non-empty keys
            if unique_non_empty_keys:
                self.metadata = list(unique_non_empty_keys)
                

        

        parameters = {
            "data": [
                {"text": text, "metadata": metadata, "embedding": embedding, "id": id}
                for text, metadata, embedding, id in zip(
                    texts, metadatas, embeddings, ids
                )
            ]
        }
        

        result = self._database.query(
                "UNWIND $data AS row "
                "CALL { "
                "    WITH row "
                f"    MERGE (c:`{self.node_label}` {{id: row.id}}) "  
                "    WITH c, row "
                f"    SET c.`{self.embedding_node_property}` = vecf32(row.embedding) "
                f"    SET c.`{self.text_node_property}` = row.text "
                "    SET c += row.metadata "
                "} "
                , params=parameters
            )


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
    
    
    @classmethod
    def from_texts(
        cls: type[FalkorDBVector],
        texts: List[str], 
        embedding: Embeddings, 
        metadatas: List[Dict] = [],
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
        ) -> FalkorDBVector:
        """
        Return FalkorDBVector initialized from texts and embeddings.
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
    
    def similarity_search(
            self,
            query: str,
            k: int = 4,
            params: Dict[str, Any] ={},
            filter: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with FalkorDBVector.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of arguments(s) to
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
        embedding =  self.embedding.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            **kwargs
        )
        return docs
 
    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            params: Dict[str, Any] = {},
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the FalkorDB database using a
        given vector and return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that
        are most similar to a given embedding. The similarity is measured
        using a vector index in the FalkorDB database. The results are returned 
        as a list of tuples, each containing a Document object and its similarity
        score.

        Args:
            embedding (List[float]): The embedding vector to compare against.
            k (int, optional): The number of top similar documents to retrieve.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to 
                    filter on metadata.
                    Defaults to None.
            params (Dict[str, Any]): The Search params for the index type.
                Defaults to empty dict.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                    a Document object and its similarity score.
        """
        index_query = _get_search_index_query(self.search_type, self._index_type)
        filter_params = {}

        if self._index_type == IndexType.RELATIONSHIP:
            if kwargs.get("return_embeddings"):
                if self.metadata:
                    # Construct the metadata part based on self.metadata
                    metadata_fields = ", ".join(f"`{key}`: relationship.`{key}`" for key in self.metadata)
                    default_retrieval = (
                        f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                        f"{{text: relationship.`{self.text_node_property}`, "
                        f"embedding: relationship.`{self.embedding_node_property}`, "
                        f"id: relationship.id, source: relationship.source, "
                        f"{metadata_fields}}} AS metadata"
                    )
                else:
                    default_retrieval = (
                        f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                        f"{{text: relationship.`{self.text_node_property}`, "
                        f"embedding: relationship.`{self.embedding_node_property}`, "
                        f"id: relationship.id, source: relationship.source}} AS metadata"
                    )
            else:
                if self.metadata:
                    # Construct the metadata part based on self.metadata
                    metadata_fields = ", ".join(f"`{key}`: relationship.`{key}`" for key in self.metadata)
                    default_retrieval = (
                        f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                        f"{{text: relationship.`{self.text_node_property}`, "
                        f"id: relationship.id, source: relationship.source, "
                        f"{metadata_fields}}} AS metadata"
                    )
                else:
                    default_retrieval = (
                        f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                        f"{{text: relationship.`{self.text_node_property}`, "
                        f"id: relationship.id, source: relationship.source}} AS metadata"
                    )
        else:
            if kwargs.get("return_embeddings"):
                if self.metadata:
                    # Construct the metadata part based on self.metadata
                    metadata_fields = ", ".join(f"`{key}`: node.`{key}`" for key in self.metadata)
                    default_retrieval = (
                        f"RETURN node.`{self.text_node_property}` AS text, score, "
                        f"{{text: node.`{self.text_node_property}`, "
                        f"embedding: node.`{self.embedding_node_property}`, "
                        f"id: node.id, source: node.source, "
                        f"{metadata_fields}}} AS metadata"
                    )
                else:
                    default_retrieval = (
                        f"RETURN node.`{self.text_node_property}` AS text, score, "
                        f"{{text: node.`{self.text_node_property}`, "
                        f"embedding: node.`{self.embedding_node_property}`, "
                        f"id: node.id, source: node.source}} AS metadata"
                    )
            else:
                if self.metadata:
                    # Construct the metadata part based on self.metadata
                    metadata_fields = ", ".join(f"`{key}`: node.`{key}`" for key in self.metadata)
                    default_retrieval = (
                        f"RETURN node.`{self.text_node_property}` AS text, score, "
                        f"{{text: node.`{self.text_node_property}`, "
                        f"id: node.id, source: node.source, "
                        f"{metadata_fields}}} AS metadata"
                    )
                else:
                    default_retrieval = (
                        f"RETURN node.`{self.text_node_property}` AS text, score, "
                        f"{{text: node.`{self.text_node_property}`, "
                        f"id: node.id, source: node.source}} AS metadata"
                    )


        retrieval_query = (
            self.retrieval_query if self.retrieval_query else default_retrieval
        )

        # print("Retrieval Query: ", retrieval_query)

        read_query = index_query + retrieval_query
        if self._index_type == "NODE":
            parameters = {
                "entity_label": self.node_label,
                "entity_property": self.embedding_node_property,
                "k": k,
                "embedding": embedding,
                "query": remove_lucene_chars(kwargs["query"]),
                **params,
                **filter_params,
            }
        elif self._index_type == "RELATIONSHIP":
            parameters = {
                "entity_label": self.relation_type,
                "entity_property": self.embedding_node_property,
                "k": k,
                "embedding": embedding,
                "query": remove_lucene_chars(kwargs["query"]),
                **params,
                **filter_params,
            }


        results = self.query(read_query, params=parameters)
        # print("RESULTS: " , results)
        

        if not results and any(result["text"] is None for result in results):
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
        #TODO: Make this property work for FalkorDB
        # Check if embeddings are missing when they are expected
        if kwargs.get("return_embeddings") and any(
            result["metadata"]["embedding"] is None for result in results
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
        try:
            docs = [
                (
                    Document(
                        page_content=result[0],  # Use the first element for text
                        metadata={k: v for k, v in result[2].items() if v is not None},  # Use the third element for metadata
                    ),
                    result[1],  # Use the second element for score
                )
                for result in results
            ]
        except AttributeError:
            try:
                docs = [
                    (
                        Document(
                            page_content=result[0],  # Use the first element for text
                            metadata={k: v for k, v in result[1].items() if v is not None},  # Use the third element for metadata
                        ),
                        result[2],  # Use the second element for score
                    )
                    for result in results
                ]
            except Exception as e:
                print(f"An error occured: {e}")
            
        #print("Docs: ", docs)
        return docs


        
    
    def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int=4,
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
            embedding=embedding, k=k, params=params, **kwargs
        )
        #print("Docs_and_scores: ", docs_and_scores)

        return[doc for doc, _ in docs_and_scores]

    
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
    ) -> FalkorDBVector:
        """Construct FalkorDBVector wrapper from raw documents and pre-
        generated embeddings.

        Return FalkorDBVector initialized from documents and embeddings.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores.falkordb_vector import FalkorDBVector
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                vectorstore = FalkorDBVector.from_embeddings(
                        text_embedding_pairs, embeddings
                )
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
        cls: Type[FalkorDBVector],
        embedding: Embeddings,
        node_label: str ,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        **kwargs: Any,
    ) -> FalkorDBVector:
        """
        Get instance of an existing FalkorDB vector index. This method will
        return the instance of the store without inserting any new 
        embeddings.
        """

        
        store = cls(
            embedding=embedding,
            node_label=node_label,
            search_type=search_type,
            **kwargs
        )



        embedding_dimension, index_type, entity_label, entity_property = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Relationship vector index is not supported with "
                "`from_existing_index` method. Please use the "
                "`from_existing_relationship_index` method."
            )

        if not index_type:
            raise ValueError(
                f"The specified vector index node label `{node_label}` on does not exist. "
                "Make sure to check if you spelled the node label correctly"
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
        cls: Type[FalkorDBVector],
        embedding: Embeddings,
        relation_type: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        **kwargs: Any,
    ) -> FalkorDBVector:
        """
        Get instance of an existing FalkorDB relationship vector index.
        This method will return the instance of the store without
        inserting any new embeddings.
        """
        if search_type == SearchType.HYBRID:
            raise ValueError(
                "Hybrid search is not supported in combination "
                "with relationship vector index"
            )
        
        store = cls(
            embedding=embedding,
            relation_type=relation_type,
            **kwargs,
        )

        embedding_dimension, index_type, entity_label, entity_property = store.retrieve_existing_relationship_index()

        if not index_type:
            raise ValueError(
                f"The specified vector index on the relationship {relation_type} does not exist. "
                "Make sure to check if you spelled it correctly"
            )
        # Raise error if not relationship index type
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
    
    
    def add_documents(
        self,
        documents: List[Document],
    ) -> List:
        """
        This function takes List[Document] element(s) and populates
        the existing store with a default node or default node(s) that 
        represents the element(s) and returns the id(s) of the newly created node(s)

        Args:
            documents: the List[Document] element
        Returns:
            a list containing the id(s) of the newly created node in the store
        """
        ids = []

        #set the index type
        self._index_type = 'NODE'

        # Add the documents to the store
        self.from_documents(
            embedding=self.embedding,
            documents=documents,
        )

        for doc in documents:
            page_content = doc.page_content
            result = self.query(
                """
                MATCH (n)
                WHERE n.text = $page_content
                RETURN n.id
                """, 
                params={"page_content": page_content}
            )
            try:
                ids.append(result[0][0])
            except Exception as e:
                raise ValueError(
                    "Your document wasn't added to the store successfully check your spellings"
                )

        return ids
            



        
        

    @classmethod
    def from_documents(
        cls: Type[FalkorDBVector],
        documents: List[Document],
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FalkorDBVector:
        """
        Return FalkorDBVector initialized from documents and embeddings.
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
        cls: Type[FalkorDBVector],
        embedding: Embeddings,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        *,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> FalkorDBVector:
        """
        Initialize and return a FalkorDBVector instance from an existing graph

        This method initializes a FalkorDBVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
            FalkorDBVector: An instance of FalkorDBVector initialized with the provided parameters
                        and existing graph.

        Example:
        >>> falkordb_vector = FalkorDBVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

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
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            **kwargs,
        )

        embedding_dimension, index_type, entity_label, entity_property = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "`from_existing_graph` method does not support "
                " existing relationship vector index. "
                "Please use `from_existing_relationship_index` method"
            )

        # If the vector index doesn't exist yet
        if not index_type:
            print(f"Making index for node label: {node_label}")
            store.create_new_index(node_label=node_label)
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index on Node {store.node_label} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        # FTS index for Hybrid search
        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
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
                f"WHERE n.`{embedding_node_property}` IS null "
                "AND any(k IN $props WHERE n[k] IS NOT null) "
                "RETURN id(n) AS id, "
                "reduce(str='', k IN $props | str + '\\n' + k + ': ' + coalesce(n[k], '')) AS text "
                "LIMIT 1000"
            )
            data = store.query(fetch_query, params={"props": text_node_properties})
            if not data:
                break
            text_embeddings = embedding.embed_documents([el[1] for el in data])

            params = {
                "data": [
                    {"id": el[0], "embedding": embedding}
                    for el, embedding in zip(data, text_embeddings)
                ]
            }

            store.query(
                "UNWIND $data AS row "
                f"MATCH (n:`{node_label}`) "
                "WHERE id(n) = row.id "
                f"SET n.`{embedding_node_property}` = vecf32(row.embedding)"
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
            #print("Got_docs: ", got_docs)
            got_embeddings = [doc.metadata["embedding"] for doc, _ in got_docs]

            # Select documents using maximal marginal relevance
            selected_indices = maximal_marginal_relevance(
                np.array(query_embedding), got_embeddings, lambda_mult=lambda_mult, k=k
            )
            selected_docs = [got_docs[i][0] for i in selected_indices]

            # Remove embedding values from metadata
            for doc in selected_docs:
                del doc.metadata["embedding"]

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
        
    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: Optional[int] = 4,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs_with_scores = self.similarity_search_with_score(
            query=query,
            k=k,
            **kwargs)
        #print("Docs_with_scores: ", docs_with_scores)

        return docs_with_scores