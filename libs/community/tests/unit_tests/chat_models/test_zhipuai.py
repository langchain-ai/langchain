import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.zhipuai import ChatZhipuAI


@pytest.mark.requires("zhipuai")
def test_chat() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chat = ChatZhipuAI(
        temperature=0.5, api_key="your_api_key", model="chatglm_turbo", streaming=False
    )
    messages = [
        AIMessage(content="Hi."),
        SystemMessage(
            content="You are a helpful assistant that translates English to Chinese."
        ),
        HumanMessage(
            content="""Translate this sentence from English to Chinese:
I love programming."""
        ),
    ]
    chat(messages)
