"""Unit tests for ChatOllamaV1."""

from langchain_core.messages.content_blocks import ImageContentBlock, TextContentBlock
from langchain_core.messages.v1 import AIMessage as AIMessageV1
from langchain_core.messages.v1 import HumanMessage as HumanMessageV1
from langchain_core.messages.v1 import MessageV1
from langchain_core.messages.v1 import SystemMessage as SystemMessageV1

from langchain_ollama._compat import (
    _convert_chunk_to_v1,
    _convert_from_v1_to_ollama_format,
    _convert_to_v1_from_ollama_format,
)
from langchain_ollama.chat_models_v1 import ChatOllamaV1


class TestMessageConversion:
    """Test v1 message conversion utilities."""

    def test_convert_human_message_v1_text_only(self) -> None:
        """Test converting HumanMessageV1 with text content."""
        message = HumanMessageV1(
            content=[TextContentBlock(type="text", text="Hello world")]
        )

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "user"
        assert result["content"] == "Hello world"
        assert result["images"] == []

    def test_convert_human_message_v1_with_image(self) -> None:
        """Test converting HumanMessageV1 with text and image content."""
        message = HumanMessageV1(
            content=[
                TextContentBlock(type="text", text="Describe this image:"),
                ImageContentBlock(  # type: ignore[typeddict-unknown-key]
                    type="image",
                    mime_type="image/jpeg",
                    data="base64imagedata",
                    source_type="base64",
                ),
            ]
        )

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "user"
        assert result["content"] == "Describe this image:"
        assert result["images"] == ["base64imagedata"]

    def test_convert_ai_message_v1(self) -> None:
        """Test converting AIMessageV1 with text content."""
        message = AIMessageV1(
            content=[TextContentBlock(type="text", text="Hello! How can I help?")]
        )

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "assistant"
        assert result["content"] == "Hello! How can I help?"

    def test_convert_system_message_v1(self) -> None:
        """Test converting SystemMessageV1."""
        message = SystemMessageV1(
            content=[TextContentBlock(type="text", text="You are a helpful assistant.")]
        )

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_convert_from_ollama_format(self) -> None:
        """Test converting Ollama response to AIMessageV1."""
        ollama_response = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 1000000,
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        result = _convert_to_v1_from_ollama_format(ollama_response)

        assert isinstance(result, AIMessageV1)
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello! How can I help you today?"
        assert result.response_metadata["model_name"] == "llama3"
        assert result.response_metadata.get("done") is True  # type: ignore[typeddict-item]

    def test_convert_chunk_to_v1(self) -> None:
        """Test converting Ollama streaming chunk to AIMessageChunkV1."""
        chunk = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        }

        result = _convert_chunk_to_v1(chunk)

        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello"

    def test_convert_empty_content(self) -> None:
        """Test converting empty content blocks."""
        message = HumanMessageV1(content=[])

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "user"
        assert result["content"] == ""
        assert result["images"] == []


class TestChatOllamaV1:
    """Test ChatOllamaV1 class."""

    def test_initialization(self) -> None:
        """Test ChatOllamaV1 initialization."""
        llm = ChatOllamaV1(model="llama3")

        assert llm.model == "llama3"
        assert llm._llm_type == "chat-ollama-v1"

    def test_chat_params(self) -> None:
        """Test _chat_params method."""
        llm = ChatOllamaV1(model="llama3", temperature=0.7)

        messages: list[MessageV1] = [
            HumanMessageV1(content=[TextContentBlock(type="text", text="Hello")])
        ]

        params = llm._chat_params(messages)

        assert params["model"] == "llama3"
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Hello"
        assert params["options"].temperature == 0.7

    def test_ls_params(self) -> None:
        """Test LangSmith parameters."""
        llm = ChatOllamaV1(model="llama3", temperature=0.5)

        ls_params = llm._get_ls_params()

        assert ls_params["ls_provider"] == "ollama"
        assert ls_params["ls_model_name"] == "llama3"
        assert ls_params["ls_model_type"] == "chat"
        assert ls_params["ls_temperature"] == 0.5

    def test_bind_tools_basic(self) -> None:
        """Test basic tool binding functionality."""
        llm = ChatOllamaV1(model="llama3")

        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Result for: {query}"

        bound_llm = llm.bind_tools([test_tool])

        # Should return a bound model
        assert bound_llm is not None
        # The actual tool binding logic is handled by the parent class
