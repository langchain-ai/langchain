import os

from qdrant_client import QdrantClient

from tests.integration_tests.fixtures import qdrant_locations


def pytest_runtest_teardown() -> None:
    """Clean up all collections after the each test."""
    for location in qdrant_locations():
        client = QdrantClient(location=location, api_key=os.getenv("QDRANT_API_KEY"))
        collections = client.get_collections().collections

        for collection in collections:
            client.delete_collection(collection.name)
