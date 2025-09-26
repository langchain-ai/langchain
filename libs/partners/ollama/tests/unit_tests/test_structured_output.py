"""Unit tests for ChatOllama structured output functionality."""

import warnings
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama


class Joke(BaseModel):
    """A simple joke schema for testing."""
    setup: str = Field(..., description="The setup of the joke")
    punchline: str = Field(..., description="The punchline of the joke")


class TestChatOllamaStructuredOutput:
    """Test ChatOllama with_structured_output functionality."""

    def test_with_structured_output_strict_mode_clean_json(self) -> None:
        """Test strict mode with clean JSON output."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return clean JSON
        mock_response = {
            "message": {"content": '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(Joke, method="json_schema", strict=True)
            result = structured_llm.invoke("Tell me a joke about chickens")

            assert isinstance(result, Joke)
            assert result.setup == "Why did the chicken cross the road?"
            assert result.punchline == "To get to the other side!"

    def test_with_structured_output_strict_mode_reasoning_prefix_fails(self) -> None:
        """Test strict mode fails with reasoning prefix."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return JSON with reasoning prefix
        mock_response = {
            "message": {"content": '<think>Let me think of a good joke...</think>{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(Joke, method="json_schema", strict=True)

            with pytest.raises(OutputParserException):
                structured_llm.invoke("Tell me a joke about chickens")

    def test_with_structured_output_lenient_mode_reasoning_prefix_succeeds(self) -> None:
        """Test lenient mode succeeds with reasoning prefix."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return JSON with reasoning prefix
        mock_response = {
            "message": {"content": '<think>Let me think of a good joke...</think>{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(Joke, method="json_schema", strict=False)
            result = structured_llm.invoke("Tell me a joke about chickens")

            assert isinstance(result, Joke)
            assert result.setup == "Why did the chicken cross the road?"
            assert result.punchline == "To get to the other side!"

    def test_with_structured_output_lenient_mode_fenced_block(self) -> None:
        """Test lenient mode with fenced code block."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return JSON in fenced block
        mock_response = {
            "message": {"content": 'Here\'s a joke for you:\n```json\n{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}\n```\nHope you like it!'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(Joke, method="json_schema", strict=False)
            result = structured_llm.invoke("Tell me a joke about chickens")

            assert isinstance(result, Joke)
            assert result.setup == "Why did the chicken cross the road?"
            assert result.punchline == "To get to the other side!"

    def test_with_structured_output_default_strict_mode(self) -> None:
        """Test that default behavior is strict mode."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return JSON with reasoning prefix
        mock_response = {
            "message": {"content": '<think>Let me think...</think>{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            # Don't specify strict parameter - should default to True
            structured_llm = llm.with_structured_output(Joke, method="json_schema")

            with pytest.raises(OutputParserException):
                structured_llm.invoke("Tell me a joke about chickens")

    def test_with_structured_output_reasoning_warning(self) -> None:
        """Test that warning is emitted when reasoning=True."""
        llm = ChatOllama(model="test-model", reasoning=True)

        # Mock the client to return clean JSON
        mock_response = {
            "message": {"content": '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                structured_llm = llm.with_structured_output(Joke, method="json_schema")

                # Check that warning was emitted
                assert len(w) == 1
                assert "Ollama reasoning is enabled" in str(w[0].message)
                assert "strict=False" in str(w[0].message)

    def test_with_structured_output_no_warning_when_reasoning_false(self) -> None:
        """Test that no warning is emitted when reasoning=False."""
        llm = ChatOllama(model="test-model", reasoning=False)

        # Mock the client to return clean JSON
        mock_response = {
            "message": {"content": '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                structured_llm = llm.with_structured_output(Joke, method="json_schema")

                # Check that no warning was emitted
                assert len(w) == 0

    def test_with_structured_output_json_mode_strict(self) -> None:
        """Test json_mode method with strict parsing."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return clean JSON
        mock_response = {
            "message": {"content": '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(Joke, method="json_mode", strict=True)
            result = structured_llm.invoke("Tell me a joke about chickens")

            assert isinstance(result, Joke)
            assert result.setup == "Why did the chicken cross the road?"
            assert result.punchline == "To get to the other side!"

    def test_with_structured_output_json_mode_lenient(self) -> None:
        """Test json_mode method with lenient parsing."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return JSON with reasoning prefix
        mock_response = {
            "message": {"content": '<think>Let me think of a joke...</think>{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(Joke, method="json_mode", strict=False)
            result = structured_llm.invoke("Tell me a joke about chickens")

            assert isinstance(result, Joke)
            assert result.setup == "Why did the chicken cross the road?"
            assert result.punchline == "To get to the other side!"

    def test_with_structured_output_dict_schema(self) -> None:
        """Test with dict schema instead of Pydantic model."""
        llm = ChatOllama(model="test-model")

        schema = {
            "type": "object",
            "properties": {
                "setup": {"type": "string"},
                "punchline": {"type": "string"}
            },
            "required": ["setup", "punchline"]
        }

        # Mock the client to return JSON with reasoning prefix
        mock_response = {
            "message": {"content": '<think>Let me think of a good joke...</think>{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(schema, method="json_schema", strict=False)
            result = structured_llm.invoke("Tell me a joke about chickens")

            assert isinstance(result, dict)
            assert result["setup"] == "Why did the chicken cross the road?"
            assert result["punchline"] == "To get to the other side!"

    def test_with_structured_output_no_json_raises_exception(self) -> None:
        """Test that no JSON content raises exception even in lenient mode."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return text without JSON
        mock_response = {
            "message": {"content": "This is just plain text with no JSON content."},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            structured_llm = llm.with_structured_output(Joke, method="json_schema", strict=False)

            with pytest.raises(OutputParserException):
                structured_llm.invoke("Tell me a joke about chickens")

    def test_with_structured_output_backward_compatibility(self) -> None:
        """Test that existing code continues to work without changes."""
        llm = ChatOllama(model="test-model")

        # Mock the client to return clean JSON
        mock_response = {
            "message": {"content": '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'},
            "done": True,
        }

        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = [mock_response]

            # Call without any new parameters - should work exactly as before
            structured_llm = llm.with_structured_output(Joke, method="json_schema")
            result = structured_llm.invoke("Tell me a joke about chickens")

            assert isinstance(result, Joke)
            assert result.setup == "Why did the chicken cross the road?"
            assert result.punchline == "To get to the other side!"
