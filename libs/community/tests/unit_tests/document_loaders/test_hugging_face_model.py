import json
from typing import Tuple

import responses
from requests import Request

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
            "region:us",
        ],
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "createdAt": "2023-12-13T21:19:59.000Z",
        "modelId": "microsoft/phi-2",
    },
    # Add additional models as needed
]

# Mocked README content for models
MOCKED_README_CONTENT = {
    "microsoft/phi-2": "README content for microsoft/phi-2",
    "openai/gpt-3": "README content for openai/gpt-3",
}


def response_callback(request: Request) -> Tuple[int, dict, str]:
    if "/api/models" in request.url:
        return (200, {}, json.dumps(MOCKED_MODELS_RESPONSE))
    elif "README.md" in request.url:
        model_id = (
            request.url.split("/")[3] + "/" + request.url.split("/")[4]
        )  # Extract model_id
        content = MOCKED_README_CONTENT.get(model_id, "")
        return (200, {}, content)
    return (404, {}, "Not Found")


@responses.activate
def test_load_models_with_readme() -> None:
    """Tests loading models along with their README content."""
    responses.add_callback(
        responses.GET,
        "https://huggingface.co/api/models",
        callback=response_callback,  # type: ignore
        content_type="application/json",
    )
    responses.add_callback(
        responses.GET,
        # Use a regex or update this placeholder
        "https://huggingface.co/microsoft/phi-2/raw/main/README.md",
        callback=response_callback,  # type: ignore
        content_type="text/plain",
    )

    loader = HuggingFaceModelLoader(search="phi-2", limit=2)
    docs = loader.load()

    assert len(docs) == len(MOCKED_MODELS_RESPONSE)
    for doc, expected_model in zip(docs, MOCKED_MODELS_RESPONSE):
        id_ = expected_model["id"]
        assert isinstance(id_, str)
        assert doc.page_content == MOCKED_README_CONTENT[id_]
        assert doc.metadata["modelId"] == expected_model["id"]
