"""Integration tests for ChatMaritaca.

These tests require a valid MARITACA_API_KEY environment variable.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

import os

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_maritaca import ChatMaritaca


@pytest.fixture
def chat_model() -> ChatMaritaca:
    """Create a ChatMaritaca instance for testing."""
    return ChatMaritaca(model="sabia-3", temperature=0.0)  # type: ignore[arg-type]


@pytest.mark.skipif(
    not os.environ.get("MARITACA_API_KEY"),
    reason="MARITACA_API_KEY not set",
)
class TestChatMaritacaIntegration:
    """Integration tests for ChatMaritaca."""

    def test_invoke_simple(self, chat_model: ChatMaritaca) -> None:
        """Test simple invoke."""
        response = chat_model.invoke([HumanMessage(content="Olá, tudo bem?")])
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    def test_invoke_with_system_message(self, chat_model: ChatMaritaca) -> None:
        """Test invoke with system message."""
        messages = [
            SystemMessage(content="Você é um assistente que responde em português."),
            HumanMessage(content="Qual é a capital do Brasil?"),
        ]
        response = chat_model.invoke(messages)
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        content_lower = response.content.lower()
        assert "brasília" in content_lower or "Brasília" in response.content

    def test_invoke_with_conversation(self, chat_model: ChatMaritaca) -> None:
        """Test invoke with multi-turn conversation."""
        messages = [
            HumanMessage(content="Meu nome é João."),
            AIMessage(content="Olá João! Como posso ajudá-lo hoje?"),
            HumanMessage(content="Qual é o meu nome?"),
        ]
        response = chat_model.invoke(messages)
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert "João" in response.content

    async def test_ainvoke_simple(self, chat_model: ChatMaritaca) -> None:
        """Test async invoke."""
        response = await chat_model.ainvoke([HumanMessage(content="Olá!")])
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    def test_stream_simple(self, chat_model: ChatMaritaca) -> None:
        """Test streaming."""
        chunks = list(chat_model.stream([HumanMessage(content="Conte até 5.")]))
        assert len(chunks) > 0
        full_response = "".join(
            str(chunk.content) for chunk in chunks if chunk.content
        )
        assert len(full_response) > 0

    async def test_astream_simple(self, chat_model: ChatMaritaca) -> None:
        """Test async streaming."""
        chunks = []
        async for chunk in chat_model.astream([HumanMessage(content="Conte até 3.")]):
            chunks.append(chunk)
        assert len(chunks) > 0
        full_response = "".join(
            str(chunk.content) for chunk in chunks if chunk.content
        )
        assert len(full_response) > 0

    def test_batch_invoke(self, chat_model: ChatMaritaca) -> None:
        """Test batch invoke."""
        messages_list: list[list[BaseMessage]] = [
            [HumanMessage(content="1 + 1 = ?")],
            [HumanMessage(content="2 + 2 = ?")],
        ]
        responses = chat_model.batch(messages_list)  # type: ignore[arg-type]
        assert len(responses) == 2
        for response in responses:
            assert isinstance(response, AIMessage)

    def test_usage_metadata(self, chat_model: ChatMaritaca) -> None:
        """Test that usage metadata is returned."""
        response = chat_model.invoke([HumanMessage(content="Olá!")])
        assert response.usage_metadata is not None
        assert response.usage_metadata["input_tokens"] > 0
        assert response.usage_metadata["output_tokens"] > 0
        assert response.usage_metadata["total_tokens"] > 0


@pytest.mark.skipif(
    not os.environ.get("MARITACA_API_KEY"),
    reason="MARITACA_API_KEY not set",
)
class TestChatMaritacaModels:
    """Test different Maritaca models."""

    def test_sabia_3(self) -> None:
        """Test sabia-3 model."""
        model = ChatMaritaca(model="sabia-3", temperature=0.0)  # type: ignore[arg-type]
        response = model.invoke([HumanMessage(content="Olá!")])
        assert isinstance(response, AIMessage)

    def test_sabiazinho_3(self) -> None:
        """Test sabiazinho-3 model."""
        model = ChatMaritaca(model="sabiazinho-3", temperature=0.0)  # type: ignore[arg-type]
        response = model.invoke([HumanMessage(content="Olá!")])
        assert isinstance(response, AIMessage)
