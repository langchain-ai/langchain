"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_groq import ChatGroq


class TestGroqStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGroq

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_standard_params(
            chat_model_class,
            chat_model_params,
        )
