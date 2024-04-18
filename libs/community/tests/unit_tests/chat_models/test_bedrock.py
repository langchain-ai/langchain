"""Test Anthropic Chat API wrapper."""

from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.chat_models import BedrockChat
from langchain_community.chat_models.meta import convert_messages_to_prompt_llama


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([HumanMessage(content="Hello")], "[INST] Hello [/INST]"),
        (
            [HumanMessage(content="Hello"), AIMessage(content="Answer:")],
            "[INST] Hello [/INST]\nAnswer:",
        ),
        (
            [
                SystemMessage(content="You're an assistant"),
                HumanMessage(content="Hello"),
                AIMessage(content="Answer:"),
            ],
            "<<SYS>> You're an assistant <</SYS>>\n[INST] Hello [/INST]\nAnswer:",
        ),
    ],
)
def test_formatting(messages: List[BaseMessage], expected: str) -> None:
    result = convert_messages_to_prompt_llama(messages)
    assert result == expected


@pytest.mark.parametrize(
    "model_id",
    ["anthropic.claude-v2", "amazon.titan-text-express-v1"],
)
def test_different_models_bedrock(model_id: str) -> None:
    provider = model_id.split(".")[0]
    client = MagicMock()
    respbody = MagicMock()
    if provider == "anthropic":
        respbody.read.return_value = MagicMock(
            decode=MagicMock(return_value=b'{"completion":"Hi back"}'),
        )
        client.invoke_model.return_value = {"body": respbody}
    elif provider == "amazon":
        respbody.read.return_value = '{"results": [{"outputText": "Hi back"}]}'
        client.invoke_model.return_value = {"body": respbody}

    model = BedrockChat(model_id=model_id, client=client)

    # should not throw an error
    model.invoke("hello there")


def test_bedrock_combine_llm_output() -> None:
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    client = MagicMock()
    llm_outputs = [
        {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3,
            },
        },
        {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3,
            },
        },
    ]
    model = BedrockChat(model_id=model_id, client=client)
    final_output = model._combine_llm_outputs(llm_outputs)  # type: ignore[arg-type]
    assert final_output["model_id"] == model_id
    assert final_output["usage"]["completion_tokens"] == 2
    assert final_output["usage"]["prompt_tokens"] == 4
    assert final_output["usage"]["total_tokens"] == 6
