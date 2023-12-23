import pytest

from langchain_community.chat_models.zhipuai import ChatZhipuAI


@pytest.mark.requires("zhipuai")
def test_integration_initialization() -> None:
    chat = ChatZhipuAI(model="chatglm_turbo", streaming=False)
    assert chat.model == "chatglm_turbo"
    assert chat.streaming is False
