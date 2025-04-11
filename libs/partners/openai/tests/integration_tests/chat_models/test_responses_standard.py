"""Standard LangChain interface tests for Responses API"""

import pytest
from langchain_core.language_models import BaseChatModel

from langchain_openai import ChatOpenAI
from tests.integration_tests.chat_models.test_base_standard import TestOpenAIStandard


class TestOpenAIResponses(TestOpenAIStandard):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini", "use_responses_api": True}

    @pytest.mark.xfail(reason="Unsupported.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)
