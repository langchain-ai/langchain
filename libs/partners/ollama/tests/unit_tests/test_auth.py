"""Test URL authentication parsing functionality."""

import base64
from unittest.mock import MagicMock, patch

from langchain_ollama._utils import parse_url_with_auth
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

MODEL_NAME = "llama3.1"


class TestParseUrlWithAuth:
    """Test the parse_url_with_auth utility function."""

    def test_parse_url_with_auth_none_input(self) -> None:
        """Test that None input returns None, None."""
        result = parse_url_with_auth(None)
        assert result == (None, None)

    def test_parse_url_with_auth_no_credentials(self) -> None:
        """Test URLs without authentication credentials."""
        url = "https://ollama.example.com:11434/path?query=param"
        result = parse_url_with_auth(url)
        assert result == (url, None)

    def test_parse_url_with_auth_no_scheme_host_port(self) -> None:
        """Test scheme-less host:port is accepted with default http scheme."""
        url = "ollama:11434"
        cleaned_url, headers = parse_url_with_auth(url)
        assert cleaned_url == "http://ollama:11434"
        assert headers is None

    def test_parse_url_with_auth_with_credentials(self) -> None:
        """Test URLs with authentication credentials."""
        url = "https://user:password@ollama.example.com:11434"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "https://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:password").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers

    def test_parse_url_with_auth_no_scheme_with_credentials(self) -> None:
        """Test scheme-less URL with userinfo credentials."""
        url = "user:password@ollama.example.com:11434"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "http://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:password").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers

    def test_parse_url_with_auth_with_path_and_query(self) -> None:
        """Test URLs with auth, path, and query parameters."""
        url = "https://user:pass@ollama.example.com:11434/api/v1?timeout=30"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "https://ollama.example.com:11434/api/v1?timeout=30"
        expected_credentials = base64.b64encode(b"user:pass").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers

    def test_parse_url_with_auth_special_characters(self) -> None:
        """Test URLs with special characters in credentials."""
        url = "https://user%40domain:p%40ssw0rd@ollama.example.com:11434"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "https://ollama.example.com:11434"
        # Note: URL parsing handles percent-encoding automatically
        expected_credentials = base64.b64encode(b"user@domain:p@ssw0rd").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers

    def test_parse_url_with_auth_only_username(self) -> None:
        """Test URLs with only username (no password)."""
        url = "https://user@ollama.example.com:11434"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "https://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers

    def test_parse_url_with_auth_empty_password(self) -> None:
        """Test URLs with empty password."""
        url = "https://user:@ollama.example.com:11434"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "https://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers


class TestChatOllamaUrlAuth:
    """Test URL authentication integration with ChatOllama."""

    @patch("langchain_ollama.chat_models.Client")
    @patch("langchain_ollama.chat_models.AsyncClient")
    def test_chat_ollama_url_auth_integration(
        self, mock_async_client: MagicMock, mock_client: MagicMock
    ) -> None:
        """Test that ChatOllama properly handles URL authentication."""
        url_with_auth = "https://user:password@ollama.example.com:11434"

        ChatOllama(
            model=MODEL_NAME,
            base_url=url_with_auth,
        )

        # Verify the clients were called with cleaned URL and auth headers
        expected_url = "https://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:password").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        mock_client.assert_called_once_with(host=expected_url, headers=expected_headers)
        mock_async_client.assert_called_once_with(
            host=expected_url, headers=expected_headers
        )

    @patch("langchain_ollama.chat_models.Client")
    @patch("langchain_ollama.chat_models.AsyncClient")
    def test_chat_ollama_url_auth_with_existing_headers(
        self, mock_async_client: MagicMock, mock_client: MagicMock
    ) -> None:
        """Test that URL auth headers merge with existing headers."""
        url_with_auth = "https://user:password@ollama.example.com:11434"
        existing_headers = {"User-Agent": "test-agent", "X-Custom": "value"}

        ChatOllama(
            model=MODEL_NAME,
            base_url=url_with_auth,
            client_kwargs={"headers": existing_headers},
        )

        # Verify headers are merged
        expected_url = "https://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:password").decode()
        expected_headers = {
            **existing_headers,
            "Authorization": f"Basic {expected_credentials}",
        }

        mock_client.assert_called_once_with(host=expected_url, headers=expected_headers)
        mock_async_client.assert_called_once_with(
            host=expected_url, headers=expected_headers
        )


class TestOllamaLLMUrlAuth:
    """Test URL authentication integration with OllamaLLM."""

    @patch("langchain_ollama.llms.Client")
    @patch("langchain_ollama.llms.AsyncClient")
    def test_ollama_llm_url_auth_integration(
        self, mock_async_client: MagicMock, mock_client: MagicMock
    ) -> None:
        """Test that OllamaLLM properly handles URL authentication."""
        url_with_auth = "https://user:password@ollama.example.com:11434"

        OllamaLLM(
            model=MODEL_NAME,
            base_url=url_with_auth,
        )

        expected_url = "https://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:password").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        mock_client.assert_called_once_with(host=expected_url, headers=expected_headers)
        mock_async_client.assert_called_once_with(
            host=expected_url, headers=expected_headers
        )


class TestOllamaEmbeddingsUrlAuth:
    """Test URL authentication integration with OllamaEmbeddings."""

    @patch("langchain_ollama.embeddings.Client")
    @patch("langchain_ollama.embeddings.AsyncClient")
    def test_ollama_embeddings_url_auth_integration(
        self, mock_async_client: MagicMock, mock_client: MagicMock
    ) -> None:
        """Test that OllamaEmbeddings properly handles URL authentication."""
        url_with_auth = "https://user:password@ollama.example.com:11434"

        OllamaEmbeddings(
            model=MODEL_NAME,
            base_url=url_with_auth,
        )

        expected_url = "https://ollama.example.com:11434"
        expected_credentials = base64.b64encode(b"user:password").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        mock_client.assert_called_once_with(host=expected_url, headers=expected_headers)
        mock_async_client.assert_called_once_with(
            host=expected_url, headers=expected_headers
        )


class TestUrlAuthEdgeCases:
    """Test edge cases and error conditions for URL authentication."""

    def test_parse_url_with_auth_malformed_url(self) -> None:
        """Test behavior with malformed URLs."""
        malformed_url = "not-a-valid-url"
        result = parse_url_with_auth(malformed_url)
        # Shouldn't return a URL as it wouldn't parse correctly or reach a server
        assert result == (None, None)

    def test_parse_url_with_auth_no_port(self) -> None:
        """Test URLs without explicit port numbers."""
        url = "https://user:password@ollama.example.com"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "https://ollama.example.com"
        expected_credentials = base64.b64encode(b"user:password").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers

    def test_parse_url_with_auth_complex_password(self) -> None:
        """Test with complex passwords containing special characters."""
        # Test password with colon, which is the delimiter
        url = "https://user:pass:word@ollama.example.com:11434"
        cleaned_url, headers = parse_url_with_auth(url)

        expected_url = "https://ollama.example.com:11434"
        # The parser should handle the first colon as the separator
        expected_credentials = base64.b64encode(b"user:pass:word").decode()
        expected_headers = {"Authorization": f"Basic {expected_credentials}"}

        assert cleaned_url == expected_url
        assert headers == expected_headers
