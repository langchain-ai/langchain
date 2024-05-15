import os

from qdrant_client import QdrantClient

from tests.integration_tests.fixtures import qdrant_locations


def pytest_sessionfinish() -> None:
    """Clean up all collections after the test session."""
    for location in qdrant_locations():
        client = QdrantClient(location=location, api_key=os.getenv("QDRANT_API_KEY"))
        collections = client.get_collections().collections

        for collection in collections:
            client.delete_collection(collection.name)
