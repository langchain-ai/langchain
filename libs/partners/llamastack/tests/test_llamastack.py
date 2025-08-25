"""Tests for LangChain Llama Stack integration."""

import pytest
from unittest.mock import Mock, patch
from langchain_llamastack import ChatLlamaStack, LlamaStackSafety
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestChatLlamaStack:
    """Test suite for ChatLlamaStack."""

    @patch('langchain_llamastack.chat_models.LlamaStackClient')
    def test_init(self, mock_client_class):
        """Test ChatLlamaStack initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        llm = ChatLlamaStack(
            model="test-model",
            base_url="http://test:8321",
            temperature=0.5,
        )

        assert llm.model == "test-model"
        assert llm.base_url == "http://test:8321"
        assert llm.temperature == 0.5
        assert llm.client == mock_client

    @patch('langchain_llamastack.chat_models.LlamaStackClient')
    def test_generate(self, mock_client_class):
        """Test basic generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.completion_message.content = "Hello, I'm a test response!"
        mock_client.inference.chat_completion.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = ChatLlamaStack(model="test-model")
        messages = [HumanMessage(content="Hello")]

        result = llm._generate(messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello, I'm a test response!"
        assert isinstance(result.generations[0].message, AIMessage)

    @patch('langchain_llamastack.chat_models.LlamaStackClient')
    def test_get_available_models(self, mock_client_class):
        """Test getting available models."""
        mock_client = Mock()
        mock_model1 = Mock()
        mock_model1.identifier = "model1"
        mock_model2 = Mock()
        mock_model2.identifier = "model2"
        mock_client.models.list.return_value = [mock_model1, mock_model2]
        mock_client_class.return_value = mock_client

        llm = ChatLlamaStack()
        models = llm.get_available_models()

        assert models == ["model1", "model2"]

    @patch('langchain_llamastack.chat_models.LlamaStackClient')
    def test_message_conversion(self, mock_client_class):
        """Test message conversion between LangChain and Llama Stack formats."""
        from langchain_llamastack.chat_models import (
            _convert_langchain_message_to_llamastack,
            _convert_llamastack_message_to_langchain
        )

        # Test LangChain to Llama Stack
        human_msg = HumanMessage(content="Hello")
        ls_msg = _convert_langchain_message_to_llamastack(human_msg)
        assert ls_msg == {"role": "user", "content": "Hello"}

        ai_msg = AIMessage(content="Hi there!")
        ls_msg = _convert_langchain_message_to_llamastack(ai_msg)
        assert ls_msg == {"role": "assistant", "content": "Hi there!"}

        system_msg = SystemMessage(content="You are helpful")
        ls_msg = _convert_langchain_message_to_llamastack(system_msg)
        assert ls_msg == {"role": "system", "content": "You are helpful"}

        # Test Llama Stack to LangChain
        ls_user_msg = {"role": "user", "content": "Hello"}
        lc_msg = _convert_llamastack_message_to_langchain(ls_user_msg)
        assert isinstance(lc_msg, HumanMessage)
        assert lc_msg.content == "Hello"

        ls_assistant_msg = {"role": "assistant", "content": "Hi!"}
        lc_msg = _convert_llamastack_message_to_langchain(ls_assistant_msg)
        assert isinstance(lc_msg, AIMessage)
        assert lc_msg.content == "Hi!"


class TestLlamaStackSafety:
    """Test suite for LlamaStackSafety."""

    @patch('langchain_llamastack.safety.LlamaStackClient')
    def test_init(self, mock_client_class):
        """Test LlamaStackSafety initialization."""
        mock_client = Mock()
        mock_shield = Mock()
        mock_shield.identifier = "test-shield"
        mock_client.shields.list.return_value = [mock_shield]
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(
            base_url="http://test:8321",
            default_shield_id="custom-shield"
        )

        assert safety.base_url == "http://test:8321"
        assert safety.default_shield_id == "custom-shield"
        assert safety.client == mock_client
        assert safety.available_shields == ["test-shield"]

    @patch('langchain_llamastack.safety.LlamaStackClient')
    def test_check_content_safe(self, mock_client_class):
        """Test content safety checking - safe content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.violation = None
        mock_client.safety.run_shield.return_value = mock_response
        mock_client.shields.list.return_value = []
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(default_shield_id="test-shield")
        result = safety.check_content("Hello world!")

        assert result.is_safe is True
        assert "safe" in result.message.lower()
        assert result.shield_id == "test-shield"

    @patch('langchain_llamastack.safety.LlamaStackClient')
    def test_check_content_unsafe(self, mock_client_class):
        """Test content safety checking - unsafe content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_violation = Mock()
        mock_violation.violation_type = "harassment"
        mock_violation.confidence = 0.9
        mock_response.violation = mock_violation
        mock_client.safety.run_shield.return_value = mock_response
        mock_client.shields.list.return_value = []
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(default_shield_id="test-shield")
        result = safety.check_content("Unsafe content")

        assert result.is_safe is False
        assert result.violation_type == "harassment"
        assert result.confidence_score == 0.9
        assert "harassment" in result.message

    @patch('langchain_llamastack.safety.LlamaStackClient')
    def test_check_conversation(self, mock_client_class):
        """Test conversation safety checking."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.violation = None
        mock_client.safety.run_shield.return_value = mock_response
        mock_client.shields.list.return_value = []
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(default_shield_id="test-shield")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        result = safety.check_conversation(messages)

        assert result.is_safe is True
        assert "safe" in result.message.lower()

    @patch('langchain_llamastack.safety.LlamaStackClient')
    def test_get_available_shields(self, mock_client_class):
        """Test getting available shields."""
        mock_client = Mock()
        mock_shield1 = Mock()
        mock_shield1.identifier = "shield1"
        mock_shield2 = Mock()
        mock_shield2.identifier = "shield2"
        mock_client.shields.list.return_value = [mock_shield1, mock_shield2]
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety()
        shields = safety.get_available_shields()

        assert shields == ["shield1", "shield2"]

    @patch('langchain_llamastack.safety.LlamaStackClient')
    def test_no_shields_available(self, mock_client_class):
        """Test behavior when no shields are available."""
        mock_client = Mock()
        mock_client.shields.list.return_value = []
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety()
        result = safety.check_content("Test content")

        assert result.is_safe is False
        assert "no shields available" in result.message.lower()


class TestSafetyResult:
    """Test suite for SafetyResult."""

    def test_safety_result_creation(self):
        """Test SafetyResult creation and methods."""
        from langchain_llamastack.safety import SafetyResult

        result = SafetyResult(
            is_safe=False,
            violation_type="harassment",
            confidence_score=0.8,
            message="Content contains harassment",
            shield_id="test-shield"
        )

        assert result.is_safe is False
        assert result.violation_type == "harassment"
        assert result.confidence_score == 0.8
        assert result.message == "Content contains harassment"
        assert result.shield_id == "test-shield"

        # Test to_dict method
        result_dict = result.to_dict()
        assert result_dict["is_safe"] is False
        assert result_dict["violation_type"] == "harassment"
        assert result_dict["confidence_score"] == 0.8

        # Test string representation
        assert "SafetyResult" in str(result)
        assert "harassment" in str(result)


class TestIntegration:
    """Integration tests."""

    @patch('langchain_llamastack.chat_models.LlamaStackClient')
    @patch('langchain_llamastack.safety.LlamaStackClient')
    def test_combined_chat_and_safety(self, mock_safety_client, mock_chat_client):
        """Test combining chat and safety functionality."""
        # Setup chat mock
        mock_chat = Mock()
        mock_chat_response = Mock()
        mock_chat_response.completion_message.content = "This is a safe response"
        mock_chat.inference.chat_completion.return_value = mock_chat_response
        mock_chat_client.return_value = mock_chat

        # Setup safety mock
        mock_safety = Mock()
        mock_safety_response = Mock()
        mock_safety_response.violation = None
        mock_safety.safety.run_shield.return_value = mock_safety_response
        mock_safety.shields.list.return_value = []
        mock_safety_client.return_value = mock_safety

        # Create instances
        llm = ChatLlamaStack(model="test-model")
        safety = LlamaStackSafety(default_shield_id="test-shield")

        # Test safe workflow
        user_input = "Hello, how are you?"

        # Check input safety
        input_result = safety.check_content(user_input)
        assert input_result.is_safe is True

        # Generate response
        response = llm.invoke([HumanMessage(content=user_input)])
        assert response.content == "This is a safe response"

        # Check output safety
        output_result = safety.check_content(response.content)
        assert output_result.is_safe is True


# Pytest fixtures and configuration
@pytest.fixture
def mock_llama_stack_client():
    """Fixture for mocked Llama Stack client."""
    with patch('langchain_llamastack.chat_models.LlamaStackClient') as mock:
        yield mock


@pytest.fixture
def mock_safety_client():
    """Fixture for mocked safety client."""
    with patch('langchain_llamastack.safety.LlamaStackClient') as mock:
        yield mock


# Test configuration
def pytest_configure(config):
    """Pytest configuration."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
