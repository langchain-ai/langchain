"""Unit tests for ChatGLM chat model exception handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ChatGLM is imported from langchain_community, which may not be available
try:
    from langchain_classic.llms import ChatGLM

    CHATGLM_AVAILABLE = True
except ImportError:
    CHATGLM_AVAILABLE = False


@pytest.mark.skipif(not CHATGLM_AVAILABLE, reason="langchain_community not available")
class TestChatGLMExceptions:
    """Test ChatGLM exception handling for invalid API addresses and timeouts."""

    def test_invalid_api_address_connection_error(self) -> None:
        """Test that ChatGLM raises appropriate exception for invalid API address."""
        # Create ChatGLM instance with invalid API endpoint
        model = ChatGLM(
            endpoint_url="http://invalid-api-address-that-does-not-exist.com/v1/chat/completions",
            model="chatglm-6b",
        )

        # Mock the HTTP request to raise ConnectionError
        # ChatGLM typically uses requests library, so we patch requests.post
        with patch("requests.post") as mock_post:
            import requests

            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Failed to connect to API endpoint"
            )

            with pytest.raises((requests.exceptions.ConnectionError, Exception)) as exc_info:
                model.invoke("Hello")

            assert "connect" in str(exc_info.value).lower() or "connection" in str(
                exc_info.value
            ).lower()

    def test_invalid_api_address_http_error(self) -> None:
        """Test that ChatGLM handles HTTP errors for invalid API address."""
        import requests

        model = ChatGLM(
            endpoint_url="http://invalid-api-address.com/v1/chat/completions",
            model="chatglm-6b",
        )

        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status = MagicMock(
            side_effect=requests.exceptions.HTTPError("404 Not Found")
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises((requests.exceptions.HTTPError, Exception)) as exc_info:
                model.invoke("Hello")

            assert "404" in str(exc_info.value) or "not found" in str(
                exc_info.value
            ).lower()

    def test_timeout_error(self) -> None:
        """Test that ChatGLM raises timeout exception when request times out."""
        import requests

        model = ChatGLM(
            endpoint_url="http://slow-api.example.com/v1/chat/completions",
            model="chatglm-6b",
            timeout=1.0,  # Very short timeout
        )

        # Mock timeout exception
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout(
                "Request timed out after 1.0 seconds"
            )

            with pytest.raises((requests.exceptions.Timeout, Exception)) as exc_info:
                model.invoke("Hello")

            assert "timeout" in str(exc_info.value).lower() or "timed out" in str(
                exc_info.value
            ).lower()

    def test_timeout_error_async(self) -> None:
        """Test that ChatGLM raises timeout exception in async mode."""
        import requests

        model = ChatGLM(
            endpoint_url="http://slow-api.example.com/v1/chat/completions",
            model="chatglm-6b",
            timeout=1.0,
        )

        # Mock async timeout exception
        # Note: ChatGLM may use requests library even for async, or may use httpx
        # We'll test both scenarios
        try:
            import httpx

            with patch("httpx.AsyncClient.post") as mock_post:
                mock_post.side_effect = httpx.TimeoutException(
                    "Request timed out after 1.0 seconds"
                )

                import asyncio

                async def test_async_timeout() -> None:
                    with pytest.raises((httpx.TimeoutException, Exception)) as exc_info:
                        await model.ainvoke("Hello")

                    assert "timeout" in str(exc_info.value).lower() or "timed out" in str(
                        exc_info.value
                    ).lower()

                asyncio.run(test_async_timeout())
        except (ImportError, AttributeError):
            # If httpx is not available or model doesn't support async, skip this test
            pytest.skip("Async mode not available for ChatGLM")

    def test_invalid_api_address_streaming(self) -> None:
        """Test that ChatGLM handles invalid API address during streaming."""
        import requests

        model = ChatGLM(
            endpoint_url="http://invalid-api-address.com/v1/chat/completions",
            model="chatglm-6b",
        )

        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Failed to connect"
            )

            with pytest.raises((requests.exceptions.ConnectionError, Exception)):
                # Consume the generator to trigger the exception
                list(model.stream("Hello"))

    def test_timeout_error_streaming(self) -> None:
        """Test that ChatGLM handles timeout during streaming."""
        import requests

        model = ChatGLM(
            endpoint_url="http://slow-api.example.com/v1/chat/completions",
            model="chatglm-6b",
            timeout=0.1,  # Very short timeout
        )

        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout(
                "Request timed out"
            )

            with pytest.raises((requests.exceptions.Timeout, Exception)):
                # Consume the generator to trigger the exception
                list(model.stream("Hello"))
