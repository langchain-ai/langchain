"""Integration tests for ChatSarvam — requires SARVAM_API_SUBSCRIPTION_KEY."""

import pytest
from langchain_core.messages import HumanMessage

from langchain_sarvamcloud.chat_models import ChatSarvam


@pytest.mark.requires("SARVAM_API_SUBSCRIPTION_KEY")
class TestChatSarvamIntegration:
    @pytest.fixture()
    def model(self) -> ChatSarvam:
        return ChatSarvam(model="sarvam-105b", temperature=0.2, max_tokens=50)

    def test_invoke_basic(self, model: ChatSarvam) -> None:
        result = model.invoke("Say hello in Hindi.")
        assert result.content
        assert isinstance(result.content, str)

    def test_invoke_with_system_message(self, model: ChatSarvam) -> None:
        from langchain_core.messages import SystemMessage

        messages = [
            SystemMessage(content="You are a helpful assistant. Reply only in Hindi."),
            HumanMessage(content="What is your name?"),
        ]
        result = model.invoke(messages)
        assert result.content

    def test_stream(self, model: ChatSarvam) -> None:
        chunks = list(model.stream("Count from 1 to 3 in Hindi."))
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks if isinstance(c.content, str))
        assert len(full) > 0

    def test_tool_calling(self) -> None:
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get weather for a location."""

            location: str = Field(..., description="City name")

        model = ChatSarvam(model="sarvam-105b", temperature=0.0)
        model_with_tools = model.bind_tools([GetWeather])
        result = model_with_tools.invoke("What is the weather in Delhi?")
        assert result.tool_calls or result.content

    def test_reasoning_effort_high(self) -> None:
        model = ChatSarvam(
            model="sarvam-105b",
            reasoning_effort="high",
            max_tokens=100,
        )
        result = model.invoke("What is 2 + 2?")
        assert result.content

    def test_usage_metadata(self, model: ChatSarvam) -> None:
        result = model.invoke("Hello")
        if result.usage_metadata:
            assert result.usage_metadata.input_tokens >= 0
            assert result.usage_metadata.output_tokens >= 0
