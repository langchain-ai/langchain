import pytest

from langchain_community.chat_models.zhipuai import ChatZhipuAI


@pytest.mark.requires("zhipuai")
def test_integration_initialization() -> None:
    chat = ChatZhipuAI(model="glm-4", streaming=False)
    assert chat.model == "glm-4"
    assert chat.streaming is False
