"""Test OCI Data Science Model Deployment Endpoint."""

import pytest
import responses
from langchain_core.messages import AIMessage, HumanMessage
from pytest_mock import MockerFixture

from langchain_community.chat_models import ChatOCIModelDeployment


@pytest.mark.requires("ads")
def test_initialization(mocker: MockerFixture) -> None:
    """Test chat model initialization."""
    mocker.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
    chat = ChatOCIModelDeployment(
        model="odsc",
        endpoint="test_endpoint",
        model_kwargs={"temperature": 0.2},
    )
    assert chat.model == "odsc"
    assert chat.endpoint == "test_endpoint"
    assert chat.model_kwargs == {"temperature": 0.2}
    assert chat._identifying_params == {
        "endpoint": chat.endpoint,
        "model_kwargs": {"temperature": 0.2},
        "model": chat.model,
        "stop": chat.stop,
        "stream": chat.streaming,
    }


@pytest.mark.requires("ads")
@responses.activate
def test_call(mocker: MockerFixture) -> None:
    """Test valid call to oci model deployment endpoint."""
    endpoint = "https://MD_OCID/predict"
    responses.add(
        responses.POST,
        endpoint,
        json={
            "id": "cmpl-88159e77c92f46088faad75fce2e26a1",
            "object": "chat.completion",
            "created": 274246,
            "model": "odsc-llm",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello World",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 20,
                "completion_tokens": 10,
            },
        },
        status=200,
    )
    mocker.patch("ads.common.auth.default_signer", return_value=dict(signer=None))

    chat = ChatOCIModelDeployment(endpoint=endpoint)
    output = chat.invoke("this is a test.")
    assert isinstance(output, AIMessage)
    assert output.response_metadata == {
        "token_usage": {
            "prompt_tokens": 10,
            "total_tokens": 20,
            "completion_tokens": 10,
        },
        "model_name": "odsc-llm",
        "system_fingerprint": "",
        "finish_reason": "stop",
    }


@pytest.mark.requires("ads")
@responses.activate
def test_construct_json_body(mocker: MockerFixture) -> None:
    """Tests constructing json body that will be sent to endpoint."""
    mocker.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
    messages = [
        HumanMessage(content="User message"),
    ]
    chat = ChatOCIModelDeployment(
        endpoint="test_endpoint", model_kwargs={"temperature": 0.2}
    )
    payload = chat._construct_json_body(messages, chat._invocation_params(stop=None))
    assert payload == {
        "messages": [{"content": "User message", "role": "user"}],
        "model": chat.model,
        "stop": None,
        "stream": chat.streaming,
        "temperature": 0.2,
    }
