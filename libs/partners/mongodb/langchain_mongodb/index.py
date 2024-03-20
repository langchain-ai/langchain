import logging
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection

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


def ensure_vector_search_index(
    collection: Collection,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: List[Dict[str, str]],
    update: bool = False,
) -> None:
    """Creates a MongoDB Atlas Vector Search Index

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding
        path (str): field with vector embedding
        similarity (str): The similarity score used for the index
        filters (List[Dict[str, str]]): additional filters for index definition.
        update (bool, optional): Update Existing Search index. Defaults to False.
    """
    index_exists = bool(list(collection.list_search_indexes(index_name)))

    if update:
        if index_exists:
            update_vector_search_index(
                collection,
                index_name,
                dimensions,
                path,
                similarity,
                filters,
            )
        else:
            logger.info(
                "Index %s does not exist."
                " will not attempt to update;"
                " consider setting update = False to create"
                " a new vector search index",
                index_name,
            )

    else:
        if not index_exists:
            create_vector_search_index(
                collection,
                index_name,
                dimensions,
                path,
                similarity,
                filters,
            )
        else:
            logger.info(
                "Index %s already exists,"
                " will not attempt to index creation;"
                " consider setting update = True to update"
                " an existing vector search index",
                index_name,
            )


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
    result = db.command(
        {"createSearchIndexes": collection.name, "indexes": [index_definition]}
    )
    logger.info(result)


def drop_vector_search_index(collection: Collection, index_name: str) -> None:
    """Drop a created vector search index

    Args:
        collection (Collection): MongoDB Collection with index to be dropped
        index_name (str): Name of the MongoDB index
    """
    if bool(list(collection.list_search_indexes(index_name))):
        collection.database.command(
            {"dropSearchIndex": collection.name, "name": index_name}
        )
    else:
        logger.info("Index Name %s not found; no delete scheduled", index_name)


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
    result = db.command(
        {
            "updateSearchIndex": collection.name,
            "name": index_name,
            "definition": _create_index_definition(
                dimensions=dimensions, path=path, similarity=similarity, filters=filters
            ),
        }
    )
    logger.info(result)
