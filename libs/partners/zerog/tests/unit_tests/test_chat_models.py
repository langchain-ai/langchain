"""Unit tests for ChatZeroG."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_zerog import ChatZeroG


class TestChatZeroG:
    """Test ChatZeroG functionality."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )
        assert llm.model == "llama-3.3-70b-instruct"
        assert llm.provider_address == "0xf07240Efa67755B5311bc75784a061eDB47165Dd"

    def test_init_with_custom_provider(self) -> None:
        """Test initialization with custom provider."""
        custom_provider = "0x1234567890123456789012345678901234567890"
        llm = ChatZeroG(
            model="custom-model",
            provider_address=custom_provider,
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )
        assert llm.provider_address == custom_provider

    def test_init_without_api_key(self) -> None:
        """Test initialization without API key raises error."""
        with pytest.raises(ValueError, match="ZEROG_PRIVATE_KEY must be set"):
            ChatZeroG(model="llama-3.3-70b-instruct")

    def test_convert_messages_to_openai_format(self) -> None:
        """Test message conversion to OpenAI format."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
        ]

        openai_messages = llm._convert_messages_to_openai_format(messages)

        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        assert openai_messages == expected

    def test_create_chat_request(self) -> None:
        """Test chat request creation."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
            temperature=0.8,
            max_tokens=100,
        )

        messages = [HumanMessage(content="Hello!")]
        request = llm._create_chat_request(messages)

        assert request["model"] == "llama-3.3-70b-instruct"
        assert request["temperature"] == 0.8
        assert request["max_tokens"] == 100
        assert request["messages"] == [{"role": "user", "content": "Hello!"}]

    @pytest.mark.asyncio
    async def test_agenerate_success(self) -> None:
        """Test successful async generation."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        # Mock the broker and its methods
        mock_broker = AsyncMock()
        mock_broker.acknowledge_provider = AsyncMock()
        mock_broker.get_service_metadata = AsyncMock(return_value={
            "endpoint": "https://api.0g.ai/v1",
            "model": "llama-3.3-70b-instruct"
        })
        mock_broker.get_request_headers = AsyncMock(return_value={
            "Authorization": "Bearer test-token",
            "X-Provider": "0xf07240Efa67755B5311bc75784a061eDB47165Dd",
        })
        mock_broker.process_response = AsyncMock(return_value=True)

        llm._broker = mock_broker

        # Mock the HTTP response
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "Hello! How can I help you today?",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            },
            "model": "llama-3.3-70b-instruct"
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.raise_for_status = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_response_data)

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            messages = [HumanMessage(content="Hello!")]
            result = await llm._agenerate(messages)

            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hello! How can I help you today?"
            assert result.generations[0].message.usage_metadata["input_tokens"] == 10
            assert result.generations[0].message.usage_metadata["output_tokens"] == 8

    @pytest.mark.asyncio
    async def test_fund_account(self) -> None:
        """Test account funding."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        mock_broker = AsyncMock()
        mock_broker.fund_account = AsyncMock(return_value={"status": "success"})
        llm._broker = mock_broker

        result = await llm.fund_account("0.1")

        assert result == {"status": "success"}
        mock_broker.fund_account.assert_called_once_with("0.1")

    @pytest.mark.asyncio
    async def test_get_balance(self) -> None:
        """Test balance retrieval."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        mock_broker = AsyncMock()
        mock_broker.get_balance = AsyncMock(return_value={
            "balance": "1.0",
            "locked": "0.1",
            "available": "0.9"
        })
        llm._broker = mock_broker

        result = await llm.get_balance()

        assert result["balance"] == "1.0"
        assert result["available"] == "0.9"

    def test_bind_tools(self) -> None:
        """Test tool binding."""
        from pydantic import BaseModel, Field

        class TestTool(BaseModel):
            """Test tool for weather."""
            location: str = Field(description="Location to get weather for")

        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        bound_llm = llm.bind_tools([TestTool])

        # Check that tools are bound (this creates a new runnable)
        assert bound_llm is not llm

    def test_with_structured_output(self) -> None:
        """Test structured output."""
        from pydantic import BaseModel, Field

        class TestOutput(BaseModel):
            """Test output structure."""
            answer: str = Field(description="The answer")
            confidence: float = Field(description="Confidence score")

        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        structured_llm = llm.with_structured_output(TestOutput)

        # Check that structured output creates a new runnable
        assert structured_llm is not llm

    def test_lc_secrets(self) -> None:
        """Test that secrets are properly configured."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        assert llm.lc_secrets == {"private_key": "ZEROG_PRIVATE_KEY"}

    def test_llm_type(self) -> None:
        """Test LLM type."""
        llm = ChatZeroG(
            model="llama-3.3-70b-instruct",
            private_key="0x1234567890abcdef1234567890abcdef12345678901234567890abcdef12345678",
        )

        assert llm._llm_type == "chat-zerog"
