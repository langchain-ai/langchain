"""Test Moonshot Chat Model."""

from typing import Type, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import SecretStr
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_community.chat_models.moonshot import MoonshotChat


class TestMoonshotChat(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return MoonshotChat

    @property
    def chat_model_params(self) -> dict:
        return {"model": "moonshot-v1-8k"}

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)


def test_chat_moonshot_instantiate_with_alias() -> None:
    """Test MoonshotChat instantiate when using alias."""
    api_key = "your-api-key"
    chat = MoonshotChat(api_key=api_key)  # type: ignore[call-arg]
    assert cast(SecretStr, chat.moonshot_api_key).get_secret_value() == api_key
