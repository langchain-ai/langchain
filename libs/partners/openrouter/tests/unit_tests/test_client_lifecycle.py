"""Unit tests for OpenRouter client lifecycle."""

from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from langchain_openrouter.chat_models import ChatOpenRouter


def test_chat_openrouter_close_closes_httpx_clients() -> None:
    model = ChatOpenRouter(
        model="anthropic/claude-sonnet-4-5",
        openrouter_api_key=SecretStr("test-key"),
        app_url="https://example.com",
    )
    sync_client = MagicMock()
    sync_client.is_closed = False
    async_client = MagicMock()
    async_client.is_closed = False
    model._sync_httpx_client = sync_client
    model._async_httpx_client = async_client

    model.close()

    sync_client.close.assert_called_once()
    assert model._sync_httpx_client is None
    assert model._async_httpx_client is None
