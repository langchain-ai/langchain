"""Unit tests for ChatOllamaV1."""

from langchain_core.messages.content_blocks import (
    create_image_block,
    create_text_block,
)
from langchain_core.messages.v1 import AIMessage as AIMessageV1
from langchain_core.messages.v1 import HumanMessage as HumanMessageV1
from langchain_core.messages.v1 import MessageV1
from langchain_core.messages.v1 import SystemMessage as SystemMessageV1
from langchain_tests.unit_tests.chat_models_v1 import ChatModelV1UnitTests

from langchain_ollama._compat import (
    _convert_chunk_to_v1,
    _convert_from_v1_to_ollama_format,
    _convert_to_v1_from_ollama_format,
)
from langchain_ollama.chat_models_v1 import ChatOllamaV1

MODEL_NAME = "llama3.1"


class TestMessageConversion:
    """Test v1 message conversion utilities."""

    def test_convert_human_message_v1_text_only(self) -> None:
        """Test converting HumanMessageV1 with text content."""
        message = HumanMessageV1("Hello world")

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "user"
        assert result["content"] == "Hello world"
        assert result["images"] == []

    def test_convert_ai_message_v1(self) -> None:
        """Test converting AIMessageV1 with text content."""
        message = AIMessageV1("Hello! How can I help?")

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "assistant"
        assert result["content"] == "Hello! How can I help?"

    def test_convert_system_message_v1(self) -> None:
        """Test converting SystemMessageV1."""
        message = SystemMessageV1("You are a helpful assistant.")

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_convert_human_message_v1_with_image(self) -> None:
        """Test converting HumanMessageV1 with text and image content.

        Each uses `_convert_from_v1_to_ollama_format` to ensure
        that the conversion handles both text and image blocks correctly. Thus, we don't
        need additional tests for other message types that also use this function.

        """
        message_a = HumanMessageV1(
            content=[
                create_text_block("Describe this image:"),
                create_image_block(base64="base64imagedata"),
            ]
        )

        result_a = _convert_from_v1_to_ollama_format(message_a)

        assert result_a["role"] == "user"
        assert result_a["content"] == "Describe this image:"
        assert result_a["images"] == ["base64imagedata"]

        # Make sure multiple images are handled correctly
        message_b = HumanMessageV1(
            content=[
                create_text_block("Describe this image:"),
                create_image_block(base64="base64imagedata"),
                create_image_block(base64="base64dataimage"),
            ]
        )

        result_b = _convert_from_v1_to_ollama_format(message_b)

        assert result_b["role"] == "user"
        assert result_b["content"] == "Describe this image:"
        assert result_b["images"] == ["base64imagedata", "base64dataimage"]

    def test_convert_from_ollama_format(self) -> None:
        """Test converting Ollama response to `AIMessageV1`."""
        ollama_response = {
            "model": MODEL_NAME,
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
        assert result.content[0].get("type") == "text"
        assert result.content[0].get("text") == "Hello! How can I help you today?"
        assert result.response_metadata.get("model_name") == MODEL_NAME
        assert result.response_metadata.get("done") is True

    def test_convert_chunk_to_v1(self) -> None:
        """Test converting Ollama streaming chunk to `AIMessageChunkV1`."""
        chunk = {
            "model": MODEL_NAME,
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        }

        result = _convert_chunk_to_v1(chunk)

        assert len(result.content) == 1
        assert result.content[0].get("type") == "text"
        assert result.content[0].get("text") == "Hello"

    def test_convert_empty_content(self) -> None:
        """Test converting empty content blocks."""
        message = HumanMessageV1(content=[])

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "user"
        assert result["content"] == ""
        assert result["images"] == []


class TestChatOllamaV1(ChatModelV1UnitTests):
    """Test `ChatOllamaV1`."""

    @property
    def chat_model_class(self) -> type[ChatOllamaV1]:
        return ChatOllamaV1

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}

    def test_initialization(self) -> None:
        """Test `ChatOllamaV1` initialization."""
        llm = ChatOllamaV1(model=MODEL_NAME)

        assert llm.model == MODEL_NAME
        assert llm._llm_type == "chat-ollama-v1"

    def test_chat_params(self) -> None:
        """Test `_chat_params()`."""
        llm = ChatOllamaV1(model=MODEL_NAME, temperature=0.7)

        messages: list[MessageV1] = [HumanMessageV1("Hello")]

        params = llm._chat_params(messages)

        assert params["model"] == MODEL_NAME
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Hello"

        # Ensure options carry over
        assert params["options"].temperature == 0.7

    def test_ls_params(self) -> None:
        """Test LangSmith parameters."""
        llm = ChatOllamaV1(model=MODEL_NAME, temperature=0.5)

        ls_params = llm._get_ls_params()

        assert ls_params.get("ls_provider") == "ollama"
        assert ls_params.get("ls_model_name") == MODEL_NAME
        assert ls_params.get("ls_model_type") == "chat"
        assert ls_params.get("ls_temperature") == 0.5

    def test_bind_tools_basic(self) -> None:
        """Test basic tool binding functionality."""
        llm = ChatOllamaV1(model=MODEL_NAME)

        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Result for: {query}"

        bound_llm = llm.bind_tools([test_tool])

        # Should return a bound model
        assert bound_llm is not None
