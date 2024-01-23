import pytest

from langchain_community.chat_models.zhipuai import ChatZhipuAI


@pytest.mark.requires("zhipuai")
def test_integration_initialization() -> None:
    chat = ChatZhipuAI(model_name="glm-3-turbo", streaming=False)
    assert chat.model_name == "glm-3-turbo"
    assert chat.streaming is False
