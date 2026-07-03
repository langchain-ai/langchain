"""Tests for HuggingFaceEndpointEmbeddings client configuration."""

from unittest.mock import MagicMock, patch

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_endpoint_embeddings_passes_headers_and_bill_to(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Custom headers and billing org are forwarded to HF inference clients."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token="hf_xxx",  # noqa: S106
        additional_headers={"X-HF-Bill-To": "my-org"},
        bill_to="my-org",
    )

    mock_inference_client.assert_called_once()
    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs["headers"] == {"X-HF-Bill-To": "my-org"}
    assert call_kwargs["bill_to"] == "my-org"

    mock_async_client.assert_called_once()
    async_call_kwargs = mock_async_client.call_args[1]
    assert async_call_kwargs["headers"] == {"X-HF-Bill-To": "my-org"}
    assert async_call_kwargs["bill_to"] == "my-org"
