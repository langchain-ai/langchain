import pytest
from unittest.mock import Mock

from langchain_community.document_loaders import HuggingFaceModelLoader

# Mocked model data to simulate an API response
MOCKED_MODELS_RESPONSE = [
    {
        "_id": "657a1fff16886e681230c05a",
        "id": "microsoft/phi-2",
        "likes": 2692,
        "private": False,
        "downloads": 546775,
        "tags": [
            "transformers",
            "safetensors",
            "phi",
            "text-generation",
            "nlp",
            "code",
            "custom_code",
            "en",
            "license:mit",
            "autotrain_compatible",
            "endpoints_compatible",
            "has_space",
            "region:us"
        ],
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "createdAt": "2023-12-13T21:19:59.000Z",
        "modelId": "microsoft/phi-2"
    },
    {
        # Assuming a second model with its own unique metadata for testing diversity
        "_id": "1234567890abcdef12345678",
        "id": "openai/gpt-3",
        "likes": 3500,
        "private": False,
        "downloads": 1025000,
        "tags": [
            "transformers",
            "text-generation",
            "nlp",
            "en",
            "license:openai",
            "autotrain_compatible",
            "has_space"
        ],
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "createdAt": "2023-11-10T15:00:00.000Z",
        "modelId": "openai/gpt-3"
    },
]

# Mocked README content 
MOCKED_README_CONTENT = {
    "microsoft/phi-2": "README content for microsoft/phi-2",
    "openai/gpt-3": "README content for openai/gpt-3",
}


@pytest.fixture
def mock_requests_get(mocker):
    """Mocks the requests.get function."""
    def side_effect(url, params=None):
        # Mock the response for fetching models
        if url.endswith("/api/models"):
            print("api\n\n\n")
            mock_response = Mock()
            mock_response.json.return_value = MOCKED_MODELS_RESPONSE
            mock_response.raise_for_status = Mock()
            return mock_response
        # Mock the response for fetching README files
        elif "raw/main/README.md" in url:
            model_id = url.split('/')[3] + '/' + url.split('/')[4] # Extract model_id from the URL
            mock_response = Mock()
            mock_response.text = MOCKED_README_CONTENT.get(model_id, "")
            mock_response.raise_for_status = Mock()
            return mock_response
        # Raise an exception for unexpected URLs
        raise ValueError(f"Unexpected URL: {url}")

    mocker.patch('requests.get', side_effect=side_effect)

@pytest.mark.usefixtures("mock_requests_get")
def test_load_models_with_readme():
    """Tests loading models along with their README content."""
    loader = HuggingFaceModelLoader(search="phi-2",limit=2)
    docs = loader.load()

    # Ensure two documents are loaded corresponding to the mocked models
    assert len(docs) == 2

    # Check the first document for expected page_content and metadata
    doc = docs[0]
    assert doc.page_content == "README content for microsoft/phi-2"
    assert doc.metadata['modelId'] == "microsoft/phi-2"

    # Check the second document for expected page_content and metadata
    doc = docs[1]
    assert doc.page_content == "README content for openai/gpt-3"
    assert doc.metadata['modelId'] == "openai/gpt-3"

if __name__ == "__main__":
    test_load_models_with_readme()
