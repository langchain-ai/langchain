import logging
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid, OperationFailure

from langchain_mongodb.utils import _FailCode

logger = logging.getLogger(__file__)


def _create_index_definition(
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    return {
        "fields": [
            {
                "numDimensions": dimensions,
                "path": path,
                "similarity": similarity,
                "type": "vector",
            },
            *(filters or []),
        ]
    }


def create_vector_search_index(
    collection: Collection,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: List[Dict[str, str]],
) -> None:
    """Experimental Utility function to create a vector search index

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding
        path (str): field with vector embedding
        similarity (str): The similarity score used for the index
        filters (List[Dict[str, str]]): additional filters for index definition.
    """
    db = collection.database
    index_definition = {
        "definition": _create_index_definition(
            dimensions=dimensions, path=path, similarity=similarity, filters=filters
        ),
        "name": index_name,
        "type": "vectorSearch",
    }

    try:
        result = db.command(
            {"createSearchIndexes": collection.name, "indexes": [index_definition]}
        )
        logger.info(result)
    except CollectionInvalid:
        logger.error(
            "Index %s already exists,"
            " will not attempt to index creation;"
            " consider setting update = True to update"
            " an existing vector search index",
            index_name,
        )


def drop_vector_search_index(collection: Collection, index_name: str) -> None:
    """Drop a created vector search index

    Args:
        collection (Collection): MongoDB Collection with index to be dropped
        index_name (str): Name of the MongoDB index
    """
    try:
        collection.database.command(
            {"dropSearchIndex": collection.name, "name": index_name}
        )
    except OperationFailure as e:
        if e.code == _FailCode.ILLEGAL_OPERATION:
            logger.error("Index Name %s not found; no delete scheduled", index_name)


def update_vector_search_index(
    collection: Collection,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: List[Dict[str, str]],
) -> None:
    """Leverages the updateSearchIndex call

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding.
        path (str): field with vector embedding.
        similarity (str): The similarity score used for the index.
        filters (List[Dict[str, str]]): additional filters for index definition.
    """
    db = collection.database

    try:
        result = db.command(
            {
                "updateSearchIndex": collection.name,
                "name": index_name,
                "definition": _create_index_definition(
                    dimensions=dimensions,
                    path=path,
                    similarity=similarity,
                    filters=filters,
                ),
            }
        )
        logger.info(result)
    except OperationFailure as e:
        if e.code == _FailCode.INDEX_NOT_FOUND:
            logger.error(
                "Index %s does not exist."
                " will not attempt to update;"
                " consider setting update = False to create"
                " a new vector search index",
                index_name,
            )
        else:
            raise
