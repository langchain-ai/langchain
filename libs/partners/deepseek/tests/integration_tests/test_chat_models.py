"""Test ChatDeepSeek chat model."""

from typing import Optional

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_deepseek.chat_models import ChatDeepSeek


class TestChatDeepSeek(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatDeepSeek]:
        return ChatDeepSeek

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "deepseek-chat",
            "temperature": 0,
        }

    @property
    def supports_json_mode(self) -> bool:
        """(bool) whether the chat model supports JSON mode."""
        return True

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)


@pytest.mark.xfail(reason="Takes > 30s to run.")
def test_reasoning_content() -> None:
    """Test reasoning content."""
    chat_model = ChatDeepSeek(model="deepseek-reasoner")
    response = chat_model.invoke("What is 3^3?")
    assert response.content
    assert response.additional_kwargs["reasoning_content"]
    raise ValueError()


@pytest.mark.xfail(reason="Takes > 30s to run.")
def test_reasoning_content_streaming() -> None:
    chat_model = ChatDeepSeek(model="deepseek-reasoner")
    full: Optional[BaseMessageChunk] = None
    for chunk in chat_model.stream("What is 3^3?"):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]
