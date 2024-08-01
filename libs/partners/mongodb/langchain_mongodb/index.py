import logging
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

logger = logging.getLogger(__file__)


def _vector_search_index_definition(
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
    logger.info("Creating Search Index %s on %s", index_name, collection.name)
    result = collection.create_search_index(
        SearchIndexModel(
            definition=_vector_search_index_definition(
                dimensions=dimensions, path=path, similarity=similarity, filters=filters
            ),
            name=index_name,
            type="vectorSearch",
        )
    )
    logger.info(result)


def drop_vector_search_index(collection: Collection, index_name: str) -> None:
    """Drop a created vector search index

    Args:
        collection (Collection): MongoDB Collection with index to be dropped
        index_name (str): Name of the MongoDB index
    """
    logger.info(
        "Dropping Search Index %s from Collection: %s", index_name, collection.name
    )
    collection.drop_search_index(index_name)
    logger.info("Vector Search index %s.%s dropped", collection.name, index_name)


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

    logger.info(
        "Updating Search Index %s from Collection: %s", index_name, collection.name
    )
    collection.update_search_index(
        name=index_name,
        definition=_vector_search_index_definition(
            dimensions=dimensions,
            path=path,
            similarity=similarity,
            filters=filters,
        ),
    )
    logger.info("Update succeeded")
