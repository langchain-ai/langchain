import pytest
import requests

from unittest.mock import MagicMock
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings


def test_hugginggface_inferenceapi_embedding_documents_init() -> None:
    """Test huggingface embeddings."""
    embedding = HuggingFaceInferenceAPIEmbeddings(api_key="abcd123")  # type: ignore[arg-type]
    assert "abcd123" not in repr(embedding)


@pytest.fixture
def embedding_component():
    return HuggingFaceInferenceAPIEmbeddings(api_key="test-key",
                                             api_url="https://api-inference.huggingface.co",
                                             model_name="BAAI/bge-large-en-v1.5")


def test_check_for_api_errors_no_error(embedding_component):
    """Test that no error is raised when the API response is valid."""
    response = MagicMock()
    response.json.return_value = {"data": [0.1, 0.2, 0.3]}  # Mock a successful response

    try:
        embedding_component._check_for_api_errors(response)
    except Exception:
        pytest.fail("Unexpected exception raised for valid response")


def test_check_for_api_errors_with_error(embedding_component):
    """Test that the correct ValueError is raised for an API error response."""
    response = MagicMock()
    response.json.return_value = {
        "error": "Input validation error: `inputs` cannot be empty",
        "error_type": "Validation"
    }

    with pytest.raises(ValueError, match=r"HuggingFace API Error \[Validation\]: Input validation error: `inputs` cannot be empty"):
        embedding_component._check_for_api_errors(response)


def test_check_for_api_errors_invalid_json(embedding_component):
    """Test that a ValueError is raised when the response contains invalid JSON."""
    response = MagicMock()
    response.json.side_effect = requests.JSONDecodeError("Invalid JSON", "", 0)  # Simulate invalid JSON

    with pytest.raises(requests.JSONDecodeError):
        embedding_component._check_for_api_errors(response)

