"""Standard LangChain interface tests."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_perplexity import ChatPerplexity


class TestPerplexityStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatPerplexity

    @property
    def chat_model_params(self) -> dict:
        return {"model": "sonar"}
    
    @property
    def has_structured_output(self) -> bool:
        """Temporary, we don't have a high enough tier of Perplexity to test structured output."""
        return False

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(reason="TODO: handle in integration.")
    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        super().test_double_messages_conversation(model)

    @pytest.mark.xfail(reason="Raises 400: Custom stop words not supported.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)
