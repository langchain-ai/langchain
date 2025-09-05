"""Unit tests for LlamaStack chat models."""

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from langchain_llamastack.chat_models import (
    _find_working_model,
    _get_provider_headers,
    _test_model_accessibility,
    check_llamastack_status,
    create_llamastack_llm,
    get_llamastack_models,
)


class TestChatModels:
    """Test cases for LlamaStack chat models."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.base_url = "http://test-server:8321"
        self.test_model = "test-model"
        self.available_models = ["model1", "model2", "model3"]

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_llamastack.chat_models.ChatOpenAI")
    def test_create_llamastack_llm_default_params(
        self, mock_chat_openai: Any, mock_check_connection: Any
    ) -> None:
        """Test creating LLM with default parameters."""
        # Mock connection check
        mock_check_connection.return_value = {
            "connected": True,
            "models": self.available_models,
        }

        # Mock ChatOpenAI
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        with patch(
            "langchain_llamastack.chat_models._find_working_model"
        ) as mock_find_model:
            mock_find_model.return_value = "model1"

            result = create_llamastack_llm()

            assert result == mock_llm
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8321/v1/openai/v1"
            assert call_kwargs["api_key"] == "not-needed"
            assert call_kwargs["model"] == "model1"

    # @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    # @patch("langchain_llamastack.chat_models.ChatOpenAI")
    # def test_create_llamastack_llm_custom_params(
    #     self, mock_chat_openai, mock_check_connection
    # ):
    #     """Test creating LLM with custom parameters."""
    #     mock_check_connection.return_value = {
    #         "connected": True,
    #         "models": self.available_models,
    #     }

    #     mock_llm = Mock()
    #     mock_chat_openai.return_value = mock_llm

    #     result = create_llamastack_llm(
    #         model=self.test_model,
    #         base_url=self.base_url,
    #         validate_model=False,
    #         temperature=0.7,
    #     )

    #     assert result == mock_llm
    #     call_kwargs = mock_chat_openai.call_args[1]
    #     assert call_kwargs["base_url"] == f"{self.base_url}/v1/openai/v1"
    #     assert call_kwargs["model"] == self.test_model
    #     assert call_kwargs["temperature"] == 0.7

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_create_llamastack_llm_connection_failed(
        self, mock_check_connection: Any
    ) -> None:
        """Test LLM creation when connection fails."""
        mock_check_connection.return_value = {
            "connected": False,
            "error": "Connection refused",
        }

        with pytest.raises(ConnectionError, match="LlamaStack not available"):
            create_llamastack_llm()

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_create_llamastack_llm_no_models(self, mock_check_connection: Any) -> None:
        """Test LLM creation when no models are available."""
        mock_check_connection.return_value = {"connected": True, "models": []}

        with pytest.raises(ValueError, match="No models available"):
            create_llamastack_llm()

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_create_llamastack_llm_chatgpt_not_available(
        self, mock_check_connection: Any
    ) -> None:
        """Test LLM creation when ChatOpenAI is not available."""
        mock_check_connection.return_value = {
            "connected": True,
            "models": self.available_models,
        }

        with patch("langchain_llamastack.chat_models.ChatOpenAI", None):
            with pytest.raises(ImportError, match="langchain-openai is required"):
                create_llamastack_llm()

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_llamastack.chat_models.ChatOpenAI")
    def test_create_llamastack_llm_model_not_found_strict(
        self, mock_chat_openai: Any, mock_check_connection: Any
    ) -> None:
        """Test LLM creation when specified model is not found (strict mode)."""
        mock_check_connection.return_value = {
            "connected": True,
            "models": self.available_models,
        }

        with pytest.raises(ValueError, match="Model 'nonexistent-model' not found"):
            create_llamastack_llm(model="nonexistent-model", auto_fallback=False)

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_llamastack.chat_models.ChatOpenAI")
    def test_create_llamastack_llm_model_fallback(
        self, mock_chat_openai: Any, mock_check_connection: Any
    ) -> None:
        """Test LLM creation with model fallback."""
        mock_check_connection.return_value = {
            "connected": True,
            "models": self.available_models,
        }

        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        with patch(
            "langchain_llamastack.chat_models._find_working_model"
        ) as mock_find_model:
            mock_find_model.return_value = "model1"

            result = create_llamastack_llm(
                model="nonexistent-model", auto_fallback=True, validate_model=False
            )

            assert result == mock_llm
            mock_find_model.assert_called_once()

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    @patch("langchain_llamastack.chat_models.ChatOpenAI")
    def test_create_llamastack_llm_with_provider_keys(
        self, mock_chat_openai: Any, mock_check_connection: Any
    ) -> None:
        """Test LLM creation with provider API keys."""
        mock_check_connection.return_value = {
            "connected": True,
            "models": ["accounts/fireworks/models/llama-v3p1-8b-instruct"],
        }

        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        provider_keys = {"fireworks": "test-api-key"}

        result = create_llamastack_llm(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            provider_api_keys=provider_keys,
            validate_model=False,
        )

        assert result == mock_llm
        call_kwargs = mock_chat_openai.call_args[1]
        assert "default_headers" in call_kwargs
        assert "X-LlamaStack-Provider-Data" in call_kwargs["default_headers"]

    def test_get_provider_headers_fireworks(self) -> None:
        """Test getting provider headers for Fireworks model."""
        model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        provider_keys = {"fireworks": "test-api-key"}

        headers = _get_provider_headers(model, provider_keys)

        assert "X-LlamaStack-Provider-Data" in headers
        provider_data = json.loads(headers["X-LlamaStack-Provider-Data"])
        assert provider_data["fireworks_api_key"] == "test-api-key"

    def test_get_provider_headers_together(self) -> None:
        """Test getting provider headers for Together model."""
        model = "together/llama-2-7b-chat"
        provider_keys = {"together": "test-together-key"}

        headers = _get_provider_headers(model, provider_keys)

        assert "X-LlamaStack-Provider-Data" in headers
        provider_data = json.loads(headers["X-LlamaStack-Provider-Data"])
        assert provider_data["together_api_key"] == "test-together-key"

    def test_get_provider_headers_openai(self) -> None:
        """Test getting provider headers for OpenAI model."""
        model = "gpt-4"
        provider_keys = {"openai": "test-openai-key"}

        headers = _get_provider_headers(model, provider_keys)

        assert "X-LlamaStack-Provider-Data" in headers
        provider_data = json.loads(headers["X-LlamaStack-Provider-Data"])
        assert provider_data["openai_api_key"] == "test-openai-key"

    def test_get_provider_headers_anthropic(self) -> None:
        """Test getting provider headers for Anthropic model."""
        model = "claude-3-sonnet"
        provider_keys = {"anthropic": "test-anthropic-key"}

        headers = _get_provider_headers(model, provider_keys)

        assert "X-LlamaStack-Provider-Data" in headers
        provider_data = json.loads(headers["X-LlamaStack-Provider-Data"])
        assert provider_data["anthropic_api_key"] == "test-anthropic-key"

    def test_get_provider_headers_groq(self) -> None:
        """Test getting provider headers for Groq model."""
        model = "llama3-groq-70b"
        provider_keys = {"groq": "test-groq-key"}

        headers = _get_provider_headers(model, provider_keys)

        assert "X-LlamaStack-Provider-Data" in headers
        provider_data = json.loads(headers["X-LlamaStack-Provider-Data"])
        assert provider_data["groq_api_key"] == "test-groq-key"

    def test_get_provider_headers_no_match(self) -> None:
        """Test getting provider headers when no provider matches."""
        model = "ollama/llama2"
        provider_keys = {"fireworks": "test-key"}

        headers = _get_provider_headers(model, provider_keys)

        assert headers == {}

    def test_get_provider_headers_env_vars(self) -> None:
        """Test getting provider headers from environment variables."""
        model = "accounts/fireworks/models/llama-v3p1-8b-instruct"

        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "env-api-key"

            headers = _get_provider_headers(model, None)

            assert "X-LlamaStack-Provider-Data" in headers
            provider_data = json.loads(headers["X-LlamaStack-Provider-Data"])
            assert provider_data["fireworks_api_key"] == "env-api-key"

    def test_find_working_model_no_validation(self) -> None:
        """Test finding working model without validation."""
        available_models = ["model1", "model2", "model3"]

        result = _find_working_model(available_models, self.base_url, validate=False)

        assert result == "model1"

    @patch("langchain_llamastack.chat_models._test_model_accessibility")
    def test_find_working_model_with_validation_success(
        self, mock_test_accessibility: Any
    ) -> None:
        """Test finding working model with validation success."""
        available_models = ["model1", "model2", "model3"]
        mock_test_accessibility.side_effect = [False, True, False]

        result = _find_working_model(available_models, self.base_url, validate=True)

        assert result == "model2"
        assert mock_test_accessibility.call_count == 2

    @patch("langchain_llamastack.chat_models._test_model_accessibility")
    def test_find_working_model_with_validation_all_fail(
        self, mock_test_accessibility: Any
    ) -> None:
        """Test finding working model when all fail validation."""
        available_models = ["model1", "model2", "model3"]
        mock_test_accessibility.return_value = False

        result = _find_working_model(available_models, self.base_url, validate=True)

        assert result == "model1"  # Returns first model as fallback

    @patch("httpx.Client")
    def test_test_model_accessibility_success(self, mock_httpx_client: Any) -> None:
        """Test model accessibility check success."""
        mock_response = Mock()
        mock_response.status_code = 200

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        result = _test_model_accessibility(self.test_model, self.base_url)

        assert result is True

    @patch("httpx.Client")
    def test_test_model_accessibility_provider_error(
        self, mock_httpx_client: Any
    ) -> None:
        """Test model accessibility check with provider error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "provider not found"

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        result = _test_model_accessibility(self.test_model, self.base_url)

        assert result is False

    @patch("httpx.Client")
    def test_test_model_accessibility_server_error(
        self, mock_httpx_client: Any
    ) -> None:
        """Test model accessibility check with server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "usage openai"

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        result = _test_model_accessibility(self.test_model, self.base_url)

        assert result is False

    @patch("httpx.Client")
    def test_test_model_accessibility_other_error(self, mock_httpx_client: Any) -> None:
        """Test model accessibility check with other HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        result = _test_model_accessibility(self.test_model, self.base_url)

        assert result is True  # Non-provider errors are considered accessible

    @patch("httpx.Client")
    def test_test_model_accessibility_exception(self, mock_httpx_client: Any) -> None:
        """Test model accessibility check with exception."""
        mock_http_client = Mock()
        mock_http_client.post.side_effect = Exception("Network error")
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        result = _test_model_accessibility(self.test_model, self.base_url)

        assert result is False

    @patch("langchain_llamastack.chat_models.list_available_models")
    def test_get_llamastack_models_success(self, mock_list_models: Any) -> None:
        """Test getting LlamaStack models successfully."""
        mock_list_models.return_value = self.available_models

        result = get_llamastack_models(self.base_url)

        assert result == self.available_models
        mock_list_models.assert_called_once_with(self.base_url, model_type="inference")

    # @patch("langchain_llamastack.chat_models.list_available_models")
    # def test_get_llamastack_models_error(self, mock_list_models):
    #     """Test getting LlamaStack models with error."""
    #     mock_list_models.side_effect = Exception("API error")

    #     result = get_llamastack_models(self.base_url)

    #     assert result == []

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_check_llamastack_status_success(self, mock_check_connection: Any) -> None:
        """Test checking LlamaStack status successfully."""
        expected_status = {
            "connected": True,
            "models": self.available_models,
            "models_count": 3,
        }
        mock_check_connection.return_value = expected_status

        result = check_llamastack_status(self.base_url)

        assert result == expected_status
        mock_check_connection.assert_called_once_with(
            self.base_url, model_type="inference"
        )

    @patch("langchain_llamastack.chat_models.check_llamastack_connection")
    def test_check_llamastack_status_failure(self, mock_check_connection: Any) -> None:
        """Test checking LlamaStack status with failure."""
        expected_status = {
            "connected": False,
            "error": "Connection refused",
            "models": [],
            "models_count": 0,
        }
        mock_check_connection.return_value = expected_status

        result = check_llamastack_status(self.base_url)

        assert result == expected_status

    def test_get_provider_headers_multiple_providers(self) -> None:
        """Test getting provider headers for model matching multiple patterns."""
        model = (
            "fireworks-together-model"  # Hypothetical model matching multiple patterns
        )
        provider_keys = {"fireworks": "fireworks-key", "together": "together-key"}

        headers = _get_provider_headers(model, provider_keys)

        assert "X-LlamaStack-Provider-Data" in headers
        provider_data = json.loads(headers["X-LlamaStack-Provider-Data"])

        # Should include both keys since model matches both patterns
        assert "fireworks_api_key" in provider_data
        assert "together_api_key" in provider_data
        assert provider_data["fireworks_api_key"] == "fireworks-key"
        assert provider_data["together_api_key"] == "together-key"
