"""Test ZhipuAI Chat API wrapper"""

import pytest

from langchain_community.chat_models.zhipuai import ChatZhipuAI


@pytest.mark.requires("httpx", "httpx_sse", "jwt")
def test_zhipuai_model_param() -> None:
    llm = ChatZhipuAI(api_key="test", model="foo")
    assert llm.model_name == "foo"
    llm = ChatZhipuAI(api_key="test", model_name="foo")
    assert llm.model_name == "foo"
