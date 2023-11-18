"""Test `Javelin AI Gateway` chat models"""

import pytest

from langchain.chat_models import ChatJavelinAIGateway
from langchain.pydantic_v1 import SecretStr


@pytest.mark.requires("javelin_sdk")
def test_api_key_is_secret_string() -> None:
    llm = ChatJavelinAIGateway(
        gateway_uri="<javelin-ai-gateway-uri>",
        route="<javelin-ai-gateway-chat-route>",
        javelin_api_key="secret-api-key",
        params={"temperature": 0.1},
    )
    assert isinstance(llm.javelin_api_key, SecretStr)
    assert llm.javelin_api_key.get_secret_value() == "secret-api-key"


@pytest.mark.requires("javelin_sdk")
def test_api_key_masked_when_passed_via_constructor() -> None:
    llm = ChatJavelinAIGateway(
        gateway_uri="<javelin-ai-gateway-uri>",
        route="<javelin-ai-gateway-chat-route>",
        javelin_api_key="secret-api-key",
        params={"temperature": 0.1},
    )

    assert str(llm.javelin_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.javelin_api_key)
    assert "secret-api-key" not in repr(llm)
