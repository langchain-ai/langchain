"""Test Nebius AI Studio embeddings."""

from typing import Generator

import pytest

from langchain_community.embeddings.nebius_ai_studio import NebiusAIStudioEmbeddings


def test_nebius_ai_studio_initialization():
    """Test initialization of Nebius AI Studio."""
    embeddings = NebiusAIStudioEmbeddings(
        nebius_api_key="test-api-key", model="BAAI/bge-en-icl"
    )
    assert embeddings.model == "BAAI/bge-en-icl"
    assert embeddings._llm_type == "nebius-ai-studio-embeddings"
    assert embeddings.nebius_api_key.get_secret_value() == "test-api-key"
