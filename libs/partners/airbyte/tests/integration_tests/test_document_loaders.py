"""Test Airbyte embeddings."""

import os
import time

from langchain_core.prompts import PromptTemplate

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


def test_load_github_with_template() -> None:
    """Test loading from GitHub with template."""
    airbyte_loader = AirbyteLoader(
        source="source-github",
        stream="issues",
        config={
            "repositories": ["airbytehq/quickstarts"],
            "credentials": {"personal_access_token": GITHUB_TOKEN},
        },
        template=PromptTemplate.from_template("### {title}\n\n{body}"),
        include_metadata=False,
    )
    documents = airbyte_loader.load()
    assert len(documents) > 0
    # make sure documents start with markdown h3
    for doc in documents:
        assert doc.page_content.startswith(
            "###"
        ), f"Document does not start with h3: {doc.page_content[:100]}"
        # make sure no metadata from include_metadata=False
        assert doc.metadata == {}


def test_lazy_load_github() -> None:
    """Test lazy loading from GitHub."""
    airbyte_loader = AirbyteLoader(
        source="source-github",
        stream="issues",
        config={
            "repositories": ["airbytehq/quickstarts"],
            "credentials": {"personal_access_token": GITHUB_TOKEN},
        },
    )
    # lazy load should be fast, iteration slow
    t0 = time.time()
    documents = airbyte_loader.lazy_load()
    assert time.time() - t0 < 1

    # make sure some documents have body in metadata, confirm iterable
    found_body = False
    for doc in documents:
        if "body" in doc.metadata and doc.metadata["body"]:
            found_body = True
            break
    assert found_body, "No documents with body found"


async def test_alazy_load_github() -> None:
    """Test async lazy loading from GitHub."""
    airbyte_loader = AirbyteLoader(
        source="source-github",
        stream="issues",
        config={
            "repositories": ["airbytehq/quickstarts"],
            "credentials": {"personal_access_token": GITHUB_TOKEN},
        },
    )
    # lazy load should be fast, iteration slow
    t0 = time.time()
    documents = airbyte_loader.alazy_load()
    assert time.time() - t0 < 1

    # make sure some documents have body in metadata, confirm iterable
    found_body = False
    async for doc in documents:
        if "body" in doc.metadata and doc.metadata["body"]:
            found_body = True
            break
    assert found_body, "No documents with body found"
