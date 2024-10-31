"""Integration tests for Nebius AI Studio embeddings."""

import pytest

from langchain_community.embeddings.nebius_ai_studio import NebiusAIStudioEmbeddings


@pytest.mark.requires(["openai", "transformers", "sentencepiece"])
def test_nebius_ai_studio_call_query() -> None:
    """Test valid call to Nebius AI Studio."""
    embeddings = NebiusAIStudioEmbeddings()
    output = embeddings.embed_query("Test query")
    assert len(output) == 4096
    assert isinstance(output, list)


@pytest.mark.requires(["openai", "transformers", "sentencepiece"])
def test_nebius_ai_studio_async_call_docs() -> None:
    """Test valid async call to Nebius AI Studio."""
    embeddings = NebiusAIStudioEmbeddings()
    documents = ["kek lol azaza"]
    output = embeddings.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 4096
    assert isinstance(output, list)


@pytest.mark.requires(["openai", "transformers", "sentencepiece"])
async def test_nebius_ai_studio_async_call_query() -> None:
    """Test valid async call to Nebius AI Studio."""
    embeddings = NebiusAIStudioEmbeddings()
    output = await embeddings.aembed_query("Test query")
    assert len(output) == 4096
    assert isinstance(output, list)
