"""Test Moonshot Chat Model."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
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
