import json
import os
from unittest import mock

import pytest

from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.pydantic_v1 import SecretStr


@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test",
        "AZURE_OPENAI_ENDPOINT": "https://oai.azure.com/",
        "OPENAI_API_VERSION": "2023-05-01",
    },
)
@pytest.mark.requires("openai")
@pytest.mark.parametrize(
    "model_name", ["gpt-4", "gpt-4-32k", "gpt-35-turbo", "gpt-35-turbo-16k"]
)
def test_model_name_set_on_chat_result_when_present_in_response(
    model_name: str,
) -> None:
    sample_response_text = f"""
    {{
        "id": "chatcmpl-7ryweq7yc8463fas879t9hdkkdf",
        "object": "chat.completion",
        "created": 1690381189,
        "model": "{model_name}",
        "choices": [
            {{
                "index": 0,
                "finish_reason": "stop",
                "message": {{
                    "role": "assistant",
                    "content": "I'm an AI assistant that can help you."
                }}
            }}
        ],
        "usage": {{
            "completion_tokens": 28,
            "prompt_tokens": 15,
            "total_tokens": 43
        }}
    }}
    """
    # convert sample_response_text to instance of Mapping[str, Any]
    sample_response = json.loads(sample_response_text)
    mock_chat = AzureChatOpenAI()
    chat_result = mock_chat._create_chat_result(sample_response)
    assert (
        chat_result.llm_output is not None
        and chat_result.llm_output["model_name"] == model_name
    )


@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_ENDPOINT": "https://oai.azure.com/",
        "OPENAI_API_VERSION": "2023-05-01",
    },
)
@pytest.mark.requires("openai")
def test_api_key_is_secret_string_and_matches_input() -> None:
    llm = AzureChatOpenAI(openai_api_key="secret-api-key")
    assert isinstance(llm.openai_api_key, SecretStr)
    assert llm.openai_api_key.get_secret_value() == "secret-api-key"


@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_ENDPOINT": "https://oai.azure.com/",
        "OPENAI_API_VERSION": "2023-05-01",
    },
)
@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_via_constructor() -> None:
    llm = AzureChatOpenAI(openai_api_key="secret-api-key")
    assert str(llm.openai_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.openai_api_key)
    assert "secret-api-key" not in repr(llm)


@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_ENDPOINT": "https://oai.azure.com/",
        "OPENAI_API_VERSION": "2023-05-01",
    },
)
@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_via_env() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENAI_API_KEY", "secret-api-key")
        llm = AzureChatOpenAI()
        assert str(llm.openai_api_key) == "**********"
        assert "secret-api-key" not in repr(llm.openai_api_key)
        assert "secret-api-key" not in repr(llm)
