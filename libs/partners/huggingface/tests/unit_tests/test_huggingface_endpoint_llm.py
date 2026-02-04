"""Unit tests for HuggingFaceEndpoint LLM (repo_id without endpoint_url).

Regression test for issue #24742: HuggingFaceEndpoint should accept repo_id
without requiring endpoint_url. When using Hugging Face Hub inference (repo_id),
endpoint_url is not needed.
"""

from unittest.mock import MagicMock, patch

from langchain_huggingface.llms import HuggingFaceEndpoint

# Placeholder for tests only; not a real secret (noqa: S105/S106).
_TEST_HF_TOKEN = "test-token-placeholder"  # noqa: S105


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_endpoint_repo_id_without_endpoint_url(
    mock_inference_client: MagicMock,
    mock_async_inference_client: MagicMock,
) -> None:
    """HuggingFaceEndpoint can be instantiated with only repo_id and task.

    Regression test for #24742: validation must not require endpoint_url
    when repo_id is provided (Hub inference mode).
    """
    mock_inference_client.return_value = MagicMock()
    mock_async_inference_client.return_value = MagicMock()

    llm = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        huggingfacehub_api_token=_TEST_HF_TOKEN,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    assert llm.model == "microsoft/Phi-3-mini-4k-instruct"
    assert llm.repo_id == "microsoft/Phi-3-mini-4k-instruct"
    assert llm.endpoint_url is None
    assert llm.task == "text-generation"
