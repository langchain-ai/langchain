"""Test ZhipuAI Chat API wrapper"""

import pytest
from langchain_core.messages import ToolMessage

from langchain_community.chat_models.zhipuai import (
    ChatZhipuAI,
    _convert_message_to_dict,
)


@pytest.mark.requires("httpx", "httpx_sse", "jwt")
def test_zhipuai_model_param() -> None:
    llm = ChatZhipuAI(api_key="test", model="foo")
    assert llm.model_name == "foo"
    llm = ChatZhipuAI(api_key="test", model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


def test__convert_message_to_dict_with_tool() -> None:
    message = ToolMessage(name="foo", content="bar", tool_call_id="abc123")
    result = _convert_message_to_dict(message)
    expected_output = {
        "name": "foo",
        "content": "bar",
        "tool_call_id": "abc123",
        "role": "tool",
    }
    assert result == expected_output
