"""Test Airbyte embeddings."""

import os

from langchain_airbyte import AirbyteLoader

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


def test_load_github() -> None:
    """Test loading from GitHub."""
    airbyte_loader = AirbyteLoader(
        source="source-github",
        stream="issues",
        config={
            "repositories": ["airbytehq/quickstarts"],
            "credentials": {"personal_access_token": GITHUB_TOKEN},
        },
    )
    documents = airbyte_loader.load()
    assert len(documents) > 0
    # make sure some documents have body in metadata
    found_body = False
    for doc in documents:
        if "body" in doc.metadata and doc.metadata["body"]:
            found_body = True
            break
    assert found_body, "No documents with body found"
