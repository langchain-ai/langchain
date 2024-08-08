"""Search index commands are only supported on Atlas Clusters >=M10"""

import os
from typing import Generator, List, Optional

import pytest
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import index

TIMEOUT = 120
DIMENSIONS = 10


@pytest.fixture
def collection() -> Generator:
    """Depending on uri, this could point to any type of cluster."""
    uri = os.environ.get("MONGODB_ATLAS_URI")
    client: MongoClient = MongoClient(uri)
    clxn = client["db"]["collection"]
    clxn.insert_one({"foo": "bar"})
    yield clxn
    clxn.drop()


def test_search_index_commands(collection: Collection) -> None:
    index_name = "vector_index"
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    filters: Optional[List[str]] = None
    wait_until_complete = TIMEOUT

    for index_info in collection.list_search_indexes():
        index.drop_vector_search_index(
            collection, index_info["name"], wait_until_complete=wait_until_complete
        )

    assert len(list(collection.list_search_indexes())) == 0

    index.create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=dimensions,
        path=path,
        similarity=similarity,
        filters=filters,
        wait_until_complete=wait_until_complete,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name

    new_similarity = "euclidean"
    index.update_vector_search_index(
        collection,
        index_name,
        DIMENSIONS,
        "embedding",
        new_similarity,
        filters=[],
        wait_until_complete=wait_until_complete,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == new_similarity

    index.drop_vector_search_index(
        collection, index_name, wait_until_complete=wait_until_complete
    )

    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 0
