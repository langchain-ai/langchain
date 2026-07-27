"""Tests for HuggingFaceEndpointEmbeddings with custom endpoint_url."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_huggingface.embeddings.huggingface_endpoint import (
    HuggingFaceEndpointEmbeddings,
    _is_huggingface_hosted_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, False),
        ("", False),
        ("http://localhost:8081/", False),
        ("http://127.0.0.1:8080", False),
        ("http://my-tgi.internal/", False),
        ("https://api.inference-api.azure-api.net/", False),
        ("https://abc.huggingface.co/inference", True),
        ("https://xyz.hf.space/", True),
    ],
)
def test_is_huggingface_hosted_url(
    url: str | None,
    expected: bool,  # noqa: FBT001
) -> None:
    """Verify Hugging Face hosted URL checking behavior."""
    assert _is_huggingface_hosted_url(url) is expected


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_local_endpoint_does_not_pass_api_key(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """With a local endpoint_url we don't pass token/api_key."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpointEmbeddings(  # type: ignore[call-arg]
        endpoint_url="http://localhost:8081/",
    )

    mock_inference_client.assert_called_once()
    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs.get("token") is None
    assert call_kwargs.get("model") == "http://localhost:8081/"

    mock_async_client.assert_called_once()
    async_call_kwargs = mock_async_client.call_args[1]
    assert async_call_kwargs.get("token") is None
    assert async_call_kwargs.get("model") == "http://localhost:8081/"


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_hosted_endpoint_keeps_api_key(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """HF-hosted endpoint_url still gets the token."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpointEmbeddings(  # type: ignore[call-arg]
        endpoint_url="https://abc.huggingface.co/inference",
        huggingfacehub_api_token="hf_xxx",  # noqa: S106
    )

    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs.get("token") == "hf_xxx"


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_mutual_exclusivity_validation(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Verify that model, repo_id, and endpoint_url are mutually exclusive."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    # Specifying both model and endpoint_url should raise ValueError
    with pytest.raises(ValueError, match="Please specify either"):
        HuggingFaceEndpointEmbeddings(  # type: ignore[call-arg]
            model="sentence-transformers/all-mpnet-base-v2",
            endpoint_url="http://localhost:8081/",
        )

    # Specifying both repo_id and endpoint_url should raise ValueError
    with pytest.raises(ValueError, match="Please specify either"):
        HuggingFaceEndpointEmbeddings(  # type: ignore[call-arg]
            repo_id="sentence-transformers/all-mpnet-base-v2",
            endpoint_url="http://localhost:8081/",
        )

    # Specifying all three should raise ValueError
    with pytest.raises(ValueError, match="Please specify either"):
        HuggingFaceEndpointEmbeddings(  # type: ignore[call-arg]
            model="sentence-transformers/all-mpnet-base-v2",
            repo_id="sentence-transformers/all-mpnet-base-v2",
            endpoint_url="http://localhost:8081/",
        )


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_sync_and_async_embedding_inference(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Test sync and async clients feature_extraction calls and result extraction."""
    import numpy as np

    # Mock responses for feature_extraction
    mock_sync_instance = MagicMock()
    mock_sync_instance.feature_extraction.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_inference_client.return_value = mock_sync_instance

    mock_async_instance = MagicMock()

    async def mock_async_feature_extraction(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.array([[0.4, 0.5, 0.6]])

    mock_async_instance.feature_extraction = mock_async_feature_extraction
    mock_async_client.return_value = mock_async_instance

    embeddings = HuggingFaceEndpointEmbeddings(  # type: ignore[call-arg]
        endpoint_url="http://localhost:8081/",
    )

    # Test embed_documents / embed_query
    sync_res = embeddings.embed_query("hello")
    assert sync_res == [0.1, 0.2, 0.3]
    mock_sync_instance.feature_extraction.assert_called_once_with(
        text=["hello"],
    )

    # Test aembed_documents / aembed_query
    import asyncio

    async def run_async_test() -> list[float]:
        return await embeddings.aembed_query("world")

    async_res = asyncio.run(run_async_test())
    assert async_res == [0.4, 0.5, 0.6]
