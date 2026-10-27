"""Tests for `HuggingFaceEndpointEmbeddings` that make no HF API calls."""

from unittest.mock import MagicMock, patch

from langchain_core.load import dumpd

from langchain_huggingface.embeddings.huggingface_endpoint import (
    HuggingFaceEndpointEmbeddings,
)


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_api_token_not_in_repr_or_serialized(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """The token must not leak via `repr`, which `dumpd` falls back to.

    `HuggingFaceEndpointEmbeddings` is not LangChain-serializable, so anything
    that prints, logs or serializes the object gets its `repr`, and the token is
    a plain `str` field. This mirrors the same fix on `HuggingFaceEndpoint`.
    """
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token="hf_secret_token",  # noqa: S106
    )

    assert "hf_secret_token" not in repr(embeddings)
    assert "hf_secret_token" not in str(dumpd(embeddings))
    # Still handed to both clients.
    assert mock_inference_client.call_args[1]["token"] == "hf_secret_token"  # noqa: S105
    assert mock_async_client.call_args[1]["token"] == "hf_secret_token"  # noqa: S105
