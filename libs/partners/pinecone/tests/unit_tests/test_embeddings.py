from typing import Any, Type
from unittest.mock import patch

import aiohttp
import pytest
from langchain_core.utils import convert_to_secret_str
from langchain_tests.unit_tests.embeddings import EmbeddingsTests

from langchain_pinecone import PineconeEmbeddings

API_KEY = convert_to_secret_str("NOT_A_VALID_KEY")
MODEL_NAME = "multilingual-e5-large"


@pytest.fixture(autouse=True)
def mock_pinecone() -> Any:
    """Mock Pinecone client for all tests."""
    with patch("langchain_pinecone.embeddings.PineconeClient") as mock:
        yield mock


class TestPineconeEmbeddingsStandard(EmbeddingsTests):
    """Standard LangChain embeddings tests."""

    @property
    def embeddings_class(self) -> Type[PineconeEmbeddings]:
        """Get the class under test."""
        return PineconeEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        """Get the parameters for initializing the embeddings model."""
        return {
            "model": MODEL_NAME,
            "pinecone_api_key": API_KEY,
        }


class TestPineconeEmbeddingsConfig:
    """Additional configuration tests for PineconeEmbeddings."""

    def test_default_config(self) -> None:
        """Test default configuration is set correctly."""
        embeddings = PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)  # type: ignore
        assert embeddings.batch_size == 96
        assert embeddings.query_params == {"input_type": "query", "truncation": "END"}
        assert embeddings.document_params == {
            "input_type": "passage",
            "truncation": "END",
        }
        assert embeddings.dimension == 1024

    def test_custom_config(self) -> None:
        """Test custom configuration overrides defaults."""
        embeddings = PineconeEmbeddings(
            model=MODEL_NAME,
            api_key=API_KEY,
            batch_size=128,
            query_params={"custom": "param"},
            document_params={"other": "param"},
        )
        assert embeddings.batch_size == 128
        assert embeddings.query_params == {"custom": "param"}
        assert embeddings.document_params == {"other": "param"}

    @pytest.mark.asyncio
    async def test_async_client_initialization(self) -> None:
        """Test async client is initialized correctly and only when needed."""
        embeddings = PineconeEmbeddings(model=MODEL_NAME, api_key=API_KEY)
        assert embeddings._async_client is None

        # Access async_client property
        client = embeddings.async_client
        assert client is not None
        assert isinstance(client, aiohttp.ClientSession)

        # Ensure headers are set correctly
        expected_headers = {
            "Api-Key": API_KEY.get_secret_value(),
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-10",
        }
        assert client._default_headers == expected_headers
