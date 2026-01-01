from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents.middleware.summarization import SummarizationMiddleware


# --- 1. Define a Mock Model for Testing ---
# This prevents us from needing real API keys (like OpenAI) for unit tests
class FakeChatModel(BaseChatModel):
    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


# --- 2. The Tests ---


def test_format_clean_history_basic():
    """Test that basic human/ai messages are formatted correctly."""
    # Use the FakeChatModel instead of "gpt-3.5-turbo"
    middleware = SummarizationMiddleware(model=FakeChatModel())

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(
            content="Hi there", response_metadata={"token_usage": 100}
        ),  # Metadata that should be stripped
    ]

    result = middleware._format_clean_history(messages)

    expected = "User: Hello\nAssistant: Hi there"
    assert result == expected


def test_format_clean_history_tool_messages():
    """Test that tool messages are handled correctly."""
    middleware = SummarizationMiddleware(model=FakeChatModel())

    messages = [
        HumanMessage(content="Search for apples"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "123"}]),
        ToolMessage(content="Found red apples", tool_call_id="123"),
    ]

    result = middleware._format_clean_history(messages)

    # We expect the tool output to be included clearly
    assert "Tool Output: Found red apples" in result


def test_format_clean_history_multimodal():
    """Test that multimodal (list) content is flattened to string."""
    middleware = SummarizationMiddleware(model=FakeChatModel())

    # Simulate a GPT-4o multimodal message
    multimodal_content = [
        {"type": "text", "text": "Look at this image"},
        {"type": "image_url", "image_url": {"url": "http://image.com"}},
    ]
    messages = [HumanMessage(content=multimodal_content)]

    result = middleware._format_clean_history(messages)

    # It should extract "Look at this image" and handle the dict gracefully
    assert "User: Look at this image" in result
