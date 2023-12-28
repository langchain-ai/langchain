import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from langchain_community.llms.huggingface_hub import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hb-token"


@pytest.mark.requires("huggingface_hub")
@patch(
    "langchain_community.llms.huggingface_hub.is_inference_client_supported",
    return_value=True,
)
def test_huggingface_hub_with_model(mock_is_inference_client_supported) -> None:
    from huggingface_hub import InferenceClient

    llm = HuggingFaceHub(model="model-1")

    assert type(llm.client) is InferenceClient
    assert llm.client.model == "model-1"
    assert llm.client.headers["authorization"] == "Bearer hb-token"

    mock_response = MagicMock()
    type(mock_response).content = PropertyMock(return_value=b"Hello, world!")

    with patch("huggingface_hub.inference._client.get_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response

        completion = llm("Hi, how are you?")

        assert completion == "Hello, world!"


@pytest.mark.requires("huggingface_hub")
@patch(
    "langchain_community.llms.huggingface_hub.is_inference_client_supported",
    return_value=True,
)
def test_huggingface_hub_with_task(mock_is_inference_client_supported) -> None:
    from huggingface_hub import InferenceClient

    llm = HuggingFaceHub(task="task-1")

    assert type(llm.client) is InferenceClient
    assert llm.client.model is None
    assert llm.client.headers["authorization"] == "Bearer hb-token"
    assert llm.task == "task-1"


@pytest.mark.requires("huggingface_hub")
@patch(
    "langchain_community.llms.huggingface_hub.is_inference_client_supported",
    return_value=False,
)
def test_huggingface_hub_param_with_inference_api(
    mock_is_inference_client_supported,
) -> None:
    from huggingface_hub import InferenceApi

    with patch("huggingface_hub.inference_api.HfApi") as mock_hfapi:
        llm = HuggingFaceHub(repo_id="model-1", task="text-generation")

        mock_hfapi.assert_called_once_with(token="hb-token")

    assert type(llm.client) is InferenceApi
    assert "model-1" in llm.client.api_url
    assert llm.client.task == "text-generation"
    assert llm.client.headers["authorization"] == "Bearer hb-token"

    llm("Hi, how are you?")


# TODO: Assert the model invocation selectes the task
