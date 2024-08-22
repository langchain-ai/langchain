"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests.chat_models import (
    ChatModelUnitTests,
)

from langchain_groq import ChatGroq


class TestGroqStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGroq

    @pytest.mark.xfail(reason="Groq does not support tool_choice='any'")
    def test_bind_tool_pydantic(self, model: BaseChatModel) -> None:
        super().test_bind_tool_pydantic(model)
