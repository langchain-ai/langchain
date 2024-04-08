"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_basetests.unit_tests import BaseUnitTests
from langchain_core.language_models import BaseChatModel

from langchain_mistralai import ChatMistralAI


class TestMistralStandard(BaseUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatMistralAI
