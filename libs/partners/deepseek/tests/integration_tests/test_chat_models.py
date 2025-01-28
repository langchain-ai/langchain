"""Test ChatDeepSeek chat model."""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_deepseek.chat_models import ChatDeepSeek


class TestChatDeepSeek(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatDeepSeek]:
        return ChatDeepSeek

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "deepseek-chat",
            "temperature": 0,
        }


def test_reasoning_content() -> None:
    """Test reasoning content."""
    chat_model = ChatDeepSeek(model="deepseek-reasoner")
    response = chat_model.invoke("What is the square root of 256256?")
    assert response.content
    assert response.additional_kwargs["reasoning_content"]
    raise ValueError()
