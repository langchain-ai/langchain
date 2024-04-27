from abc import ABC, abstractmethod
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool


class Person(BaseModel):
    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")


@tool
def my_adder_tool(a: int, b: int) -> int:
    """Takes two integers, a and b, and returns their sum."""
    return a + b


class ChatModelUnitTests(ABC):
    @abstractmethod
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        ...

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {}

    @pytest.fixture
    def chat_model_has_tool_calling(
        self, chat_model_class: Type[BaseChatModel]
    ) -> bool:
        return chat_model_class.bind_tools is not BaseChatModel.bind_tools

    @pytest.fixture
    def chat_model_has_structured_output(
        self, chat_model_class: Type[BaseChatModel]
    ) -> bool:
        return (
            chat_model_class.with_structured_output
            is not BaseChatModel.with_structured_output
        )

    def test_chat_model_init(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        assert model is not None

    def test_chat_model_init_api_key(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        params = {**chat_model_params, "api_key": "test"}
        model = chat_model_class(**params)  # type: ignore
        assert model is not None

    def test_chat_model_init_streaming(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(streaming=True, **chat_model_params)  # type: ignore
        assert model is not None

    def test_chat_model_bind_tool_pydantic(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
        chat_model_has_tool_calling: bool,
    ) -> None:
        if not chat_model_has_tool_calling:
            return

        model = chat_model_class(**chat_model_params)

        assert hasattr(model, "bind_tools")
        tool_model = model.bind_tools([Person])
        assert tool_model is not None

    def test_chat_model_with_structured_output(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
        chat_model_has_structured_output: bool,
    ) -> None:
        if not chat_model_has_structured_output:
            return

        model = chat_model_class(**chat_model_params)
        assert model is not None
        assert model.with_structured_output(Person) is not None
