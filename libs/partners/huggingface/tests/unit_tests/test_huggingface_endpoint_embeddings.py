"""Tests for `HuggingFaceEndpointEmbeddings` (no HF API calls)."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_huggingface.embeddings.huggingface_endpoint import (
    DEFAULT_MODEL,
    HuggingFaceEndpointEmbeddings,
)

FAKE_TOKEN = "hf_xxx"  # noqa: S105


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_endpoint_url_is_used_for_both_clients(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """A self-hosted endpoint URL is what both clients talk to."""
    endpoint_url = "http://localhost:8081"

    embeddings = HuggingFaceEndpointEmbeddings(endpoint_url=endpoint_url)

    assert mock_inference_client.call_args[1]["model"] == endpoint_url
    assert mock_async_client.call_args[1]["model"] == endpoint_url

    # The endpoint identifies itself, so no repo ID is invented for it.
    assert embeddings.model is None
    assert embeddings.repo_id is None


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_self_hosted_endpoint_is_not_given_the_configured_token(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured token is not passed to the client for a non-HF host."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    HuggingFaceEndpointEmbeddings(endpoint_url="http://localhost:8081")

    assert mock_inference_client.call_args[1]["token"] is None
    assert mock_async_client.call_args[1]["token"] is None


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_hosted_endpoint_keeps_token(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """An HF-hosted endpoint URL still gets the configured token."""
    HuggingFaceEndpointEmbeddings(
        endpoint_url="https://abc.endpoints.huggingface.co",
        huggingfacehub_api_token=FAKE_TOKEN,
    )

    assert mock_inference_client.call_args[1]["token"] == FAKE_TOKEN
    assert mock_async_client.call_args[1]["token"] == FAKE_TOKEN


@pytest.mark.parametrize(
    "conflicting",
    [
        pytest.param({"model": DEFAULT_MODEL}, id="model"),
        pytest.param({"repo_id": DEFAULT_MODEL}, id="repo_id"),
    ],
)
@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_endpoint_url_conflicts_are_rejected(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
    conflicting: dict[str, Any],
) -> None:
    """`endpoint_url` is mutually exclusive with `model` and `repo_id`."""
    with pytest.raises(ValueError, match="not both"):
        HuggingFaceEndpointEmbeddings(
            endpoint_url="http://localhost:8081", **conflicting
        )


@pytest.mark.parametrize(
    "url_kwargs",
    [
        pytest.param({"model": "http://localhost:8081"}, id="model"),
        pytest.param({"repo_id": "http://localhost:8081"}, id="repo_id"),
    ],
)
@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_url_in_repo_id_field_points_at_endpoint_url(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
    url_kwargs: dict[str, Any],
) -> None:
    """Rejecting a URL should say which parameter to use instead."""
    with pytest.raises(ValueError, match="endpoint_url"):
        HuggingFaceEndpointEmbeddings(**url_kwargs)


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_defaults_are_unchanged_without_endpoint_url(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Regression guard: the repo ID path is untouched by the new parameter."""
    embeddings = HuggingFaceEndpointEmbeddings()

    assert embeddings.model == DEFAULT_MODEL
    assert embeddings.repo_id == DEFAULT_MODEL
    assert mock_inference_client.call_args[1]["model"] == DEFAULT_MODEL


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_model_and_repo_id_together_still_accepted(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Regression guard: passing both keeps working, with `model` taking priority."""
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        repo_id="some/other-model",
    )

    assert embeddings.model == "sentence-transformers/all-mpnet-base-v2"
    assert embeddings.repo_id == "sentence-transformers/all-mpnet-base-v2"


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_embed_query_uses_endpoint_client(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Embedding calls go through the client built for the endpoint URL."""
    response = MagicMock()
    response.tolist.return_value = [[0.1, 0.2, 0.3]]
    mock_inference_client.return_value.feature_extraction.return_value = response

    embeddings = HuggingFaceEndpointEmbeddings(endpoint_url="http://localhost:8081")

    assert embeddings.embed_query("hello") == [0.1, 0.2, 0.3]
    mock_inference_client.return_value.feature_extraction.assert_called_once_with(
        text=["hello"]
    )


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
async def test_aembed_query_uses_endpoint_client(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """The async client is wired to the endpoint URL too."""
    response = MagicMock()
    response.tolist.return_value = [[0.4, 0.5]]

    async def fake_feature_extraction(**_kwargs: object) -> MagicMock:
        return response

    mock_async_client.return_value.feature_extraction = fake_feature_extraction

    embeddings = HuggingFaceEndpointEmbeddings(endpoint_url="http://localhost:8081")

    assert await embeddings.aembed_query("hello") == [0.4, 0.5]
