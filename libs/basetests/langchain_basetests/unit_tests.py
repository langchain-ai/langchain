from abc import ABC, abstractmethod
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel


class BaseUnitTests(ABC):
    @abstractmethod
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        ...

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {}

    def test_init_chat_model(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        assert model is not None
