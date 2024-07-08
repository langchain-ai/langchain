"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_together import ChatTogether


class TestTogetherStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatTogether

    @property
    def chat_model_params(self) -> dict:
        return {"model": "mistralai/Mistral-7B-Instruct-v0.1"}

    @pytest.mark.xfail(reason=("May not call a tool."))
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)
