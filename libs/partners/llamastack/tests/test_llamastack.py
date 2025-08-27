"""Tests for LangChain Llama Stack integration."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_llamastack import (
    check_llamastack_status,
    create_llamastack_llm,
    get_llamastack_models,
    LlamaStackSafety,
)


class TestFactoryFunctions:
    """Test suite for new factory functions."""

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_openai.ChatOpenAI")
    def test_create_llamastack_llm_basic(self, mock_chatopenaai, mock_check):
        """Test basic create_llamastack_llm usage."""
        # Mock successful connection with models
        mock_check.return_value = {
            "connected": True,
            "models": ["llama3.1:8b", "llama3:70b"],
            "models_count": 2,
        }

        mock_llm = Mock()
        mock_chatopenaai.return_value = mock_llm

        llm = create_llamastack_llm()

        # Should auto-select first model
        mock_chatopenaai.assert_called_once_with(
            base_url="http://localhost:8321/v1/openai/v1",
            api_key="not-needed",
            model="llama3.1:8b",
        )
        assert llm == mock_llm

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_openai.ChatOpenAI")
    def test_create_llamastack_llm_with_model(self, mock_chatopenaai, mock_check):
        """Test create_llamastack_llm with specific model."""
        mock_check.return_value = {
            "connected": True,
            "models": ["llama3.1:8b", "llama3:70b"],
            "models_count": 2,
        }

        mock_llm = Mock()
        mock_chatopenaai.return_value = mock_llm

        llm = create_llamastack_llm(model="llama3:70b", temperature=0.7)

        mock_chatopenaai.assert_called_once_with(
            base_url="http://localhost:8321/v1/openai/v1",
            api_key="not-needed",
            model="llama3:70b",
            temperature=0.7,
        )

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_create_llamastack_llm_connection_failed(self, mock_check):
        """Test create_llamastack_llm when connection fails."""
        mock_check.return_value = {
            "connected": False,
            "error": "Connection refused",
            "models": [],
        }

        with pytest.raises(ConnectionError) as exc_info:
            create_llamastack_llm()

        assert "LlamaStack not available" in str(exc_info.value)

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_create_llamastack_llm_no_models(self, mock_check):
        """Test create_llamastack_llm when no models available."""
        mock_check.return_value = {"connected": True, "models": [], "models_count": 0}

        with pytest.raises(ValueError) as exc_info:
            create_llamastack_llm()

        assert "No models available" in str(exc_info.value)

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_openai.ChatOpenAI")
    def test_create_llamastack_llm_auto_fallback(self, mock_chatopenaai, mock_check):
        """Test create_llamastack_llm with auto-fallback."""
        mock_check.return_value = {
            "connected": True,
            "models": ["llama3.1:8b", "llama3:70b"],
            "models_count": 2,
        }

        mock_llm = Mock()
        mock_chatopenaai.return_value = mock_llm

        # Request non-existent model with auto-fallback
        llm = create_llamastack_llm(model="non-existent-model", auto_fallback=True)

        # Should fallback to first available model
        mock_chatopenaai.assert_called_once_with(
            base_url="http://localhost:8321/v1/openai/v1",
            api_key="not-needed",
            model="llama3.1:8b",
        )

    @patch("langchain_llamastack.chat_models.list_available_models")
    def test_get_llamastack_models(self, mock_list):
        """Test get_llamastack_models function."""
        mock_list.return_value = ["model1", "model2", "model3"]

        models = get_llamastack_models("http://test:8321")

        assert models == ["model1", "model2", "model3"]
        mock_list.assert_called_once_with("http://test:8321")

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_check_llamastack_status(self, mock_check):
        """Test check_llamastack_status function."""
        expected_status = {
            "connected": True,
            "models": ["model1", "model2"],
            "models_count": 2,
            "base_url": "http://test:8321",
        }
        mock_check.return_value = expected_status

        status = check_llamastack_status("http://test:8321")

        assert status == expected_status
        mock_check.assert_called_once_with("http://test:8321")


# ChatLlamaStack class has been removed from the project


class TestLlamaStackSafety:
    """Test suite for LlamaStackSafety."""

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_init(self, mock_client_class):
        """Test LlamaStackSafety initialization."""
        mock_client = Mock()
        mock_shield = Mock()
        mock_shield.identifier = "test-shield"
        mock_client.shields.list.return_value = [mock_shield]
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(
            base_url="http://test:8321", default_shield_id="custom-shield"
        )

        assert safety.base_url == "http://test:8321"
        assert safety.default_shield_id == "custom-shield"
        assert safety.client == mock_client
        assert safety.available_shields == ["test-shield"]

    @patch("langchain_llamastack.safety.LlamaStackClient")
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

    @patch("langchain_llamastack.safety.LlamaStackClient")
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

    @patch("langchain_llamastack.safety.LlamaStackClient")
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
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = safety.check_conversation(messages)

        assert result.is_safe is True
        assert "safe" in result.message.lower()

    @patch("langchain_llamastack.safety.LlamaStackClient")
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

    @patch("langchain_llamastack.safety.LlamaStackClient")
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
            shield_id="test-shield",
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

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_openai.ChatOpenAI")
    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_combined_chat_and_safety(
        self, mock_safety_client, mock_chat_openai, mock_check
    ):
        """Test combining chat and safety functionality."""
        # Setup connection mock
        mock_check.return_value = {
            "connected": True,
            "models": ["test-model"],
            "models_count": 1,
        }

        # Setup ChatOpenAI mock
        mock_chat_response = Mock()
        mock_chat_response.content = "This is a safe response"
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_chat_response
        mock_chat_openai.return_value = mock_llm

        # Setup safety mock
        mock_safety = Mock()
        mock_safety_response = Mock()
        mock_safety_response.violation = None
        mock_safety.safety.run_shield.return_value = mock_safety_response
        mock_safety.shields.list.return_value = []
        mock_safety_client.return_value = mock_safety

        # Create instances
        llm = create_llamastack_llm(model="test-model")
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
    with patch("langchain_llamastack.chat_models.LlamaStackClient") as mock:
        yield mock


@pytest.fixture
def mock_safety_client():
    """Fixture for mocked safety client."""
    with patch("langchain_llamastack.safety.LlamaStackClient") as mock:
        yield mock


# Test configuration
def pytest_configure(config):
    """Pytest configuration."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
