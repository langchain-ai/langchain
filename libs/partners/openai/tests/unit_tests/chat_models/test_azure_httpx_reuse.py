"""Test that Azure OpenAI classes reuse httpx clients instead of creating new ones."""

import pytest


def test_azure_chat_openai_reuses_httpx_client() -> None:
    """Test that multiple AzureChatOpenAI instances share the same httpx client."""
    from langchain_openai import AzureChatOpenAI

    # Create instances without actually connecting
    llm1 = AzureChatOpenAI(
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="gpt-4",
        api_version="2024-05-01-preview",
        api_key="test-key",
    )
    llm2 = AzureChatOpenAI(
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="gpt-4",
        api_version="2024-05-01-preview",
        api_key="test-key",
    )

    # Validate environments to trigger client creation
    llm1.validate_environment()
    llm2.validate_environment()

    # Both should have clients
    assert llm1.root_client is not None
    assert llm2.root_client is not None

    # The httpx clients should be the same cached instance
    assert llm1.root_client._client is llm2.root_client._client, (
        "AzureChatOpenAI instances should share the same httpx client "
        "to avoid resource leaks"
    )


def test_azure_chat_openai_async_reuses_httpx_client() -> None:
    """Test that multiple async instances share the same httpx client."""
    from langchain_openai import AzureChatOpenAI

    llm1 = AzureChatOpenAI(
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="gpt-4",
        api_version="2024-05-01-preview",
        api_key="test-key",
    )
    llm2 = AzureChatOpenAI(
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="gpt-4",
        api_version="2024-05-01-preview",
        api_key="test-key",
    )

    llm1.validate_environment()
    llm2.validate_environment()

    assert llm1.root_async_client is not None
    assert llm2.root_async_client is not None

    assert llm1.root_async_client._client is llm2.root_async_client._client, (
        "AzureChatOpenAI async instances should share the same httpx client"
    )


def test_azure_chat_openai_custom_http_client_not_overridden() -> None:
    """Test that a user-provided http_client is respected."""
    import httpx
    from langchain_openai import AzureChatOpenAI

    custom_client = httpx.Client(timeout=60.0)
    llm = AzureChatOpenAI(
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="gpt-4",
        api_version="2024-05-01-preview",
        api_key="test-key",
        http_client=custom_client,
    )
    llm.validate_environment()

    # Should use the custom client, not the cached one
    assert llm.root_client._client is custom_client
