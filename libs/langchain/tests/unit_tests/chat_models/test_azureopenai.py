import json
import os
from typing import Any, Mapping, cast
from unittest import mock

import pytest

from langchain.chat_models.azure_openai import AzureChatOpenAI


@mock.patch.dict(
    os.environ,
    {
        "OPENAI_API_KEY": "test",
        "OPENAI_API_BASE": "https://oai.azure.com/",
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
    mock_response = cast(Mapping[str, Any], sample_response)
    mock_chat = AzureChatOpenAI()
    chat_result = mock_chat._create_chat_result(mock_response)
    assert (
        chat_result.llm_output is not None
        and chat_result.llm_output["model_name"] == model_name
    )
