"""Unit tests for chat models."""
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool

from langchain_standard_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION


class Person(BaseModel):  # Used by some dependent tests. Should be deprecated.
    """Record attributes of a person."""

    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")


def generate_schema_pydantic_v1_from_2() -> Any:
    """Use to generate a schema from v1 namespace in pydantic 2."""
    if PYDANTIC_MAJOR_VERSION != 2:
        raise AssertionError("This function is only compatible with Pydantic v2.")
    from pydantic.v1 import BaseModel, Field

    class PersonB(BaseModel):
        """Record attributes of a person."""

        name: str = Field(..., description="The name of the person.")
        age: int = Field(..., description="The age of the person.")

    return PersonB


def generate_schema_pydantic() -> Any:
    """Works with either pydantic 1 or 2"""
    from pydantic import BaseModel as BaseModelProper
    from pydantic import Field as FieldProper

    class PersonA(BaseModelProper):
        """Record attributes of a person."""

        name: str = FieldProper(..., description="The name of the person.")
        age: int = FieldProper(..., description="The age of the person.")

    return PersonA


TEST_PYDANTIC_MODELS = [generate_schema_pydantic()]

if PYDANTIC_MAJOR_VERSION == 2:
    TEST_PYDANTIC_MODELS.append(generate_schema_pydantic_v1_from_2())


@tool
def my_adder_tool(a: int, b: int) -> int:
    """Takes two integers, a and b, and returns their sum."""
    return a + b


def my_adder(a: int, b: int) -> int:
    """Takes two integers, a and b, and returns their sum."""
    return a + b


class ChatModelTests(ABC):
    @property
    @abstractmethod
    def chat_model_class(self) -> Type[BaseChatModel]:
        ...

    @property
    def chat_model_params(self) -> dict:
        return {}

    @property
    def standard_chat_model_params(self) -> dict:
        return {
            "temperature": 0,
            "max_tokens": 100,
            "timeout": 60,
            "stop": [],
            "max_retries": 2,
        }

    @pytest.fixture
    def model(self) -> BaseChatModel:
        return self.chat_model_class(
            **{**self.standard_chat_model_params, **self.chat_model_params}
        )

    @property
    def has_tool_calling(self) -> bool:
        return self.chat_model_class.bind_tools is not BaseChatModel.bind_tools

    @property
    def has_structured_output(self) -> bool:
        return (
            self.chat_model_class.with_structured_output
            is not BaseChatModel.with_structured_output
        )

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def supports_video_inputs(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False


class ChatModelUnitTests(ChatModelTests):
    @property
    def standard_chat_model_params(self) -> dict:
        params = super().standard_chat_model_params
        params["api_key"] = "test"
        return params

    def test_init(self) -> None:
        model = self.chat_model_class(
            **{**self.standard_chat_model_params, **self.chat_model_params}
        )
        assert model is not None

    def test_init_streaming(
        self,
    ) -> None:
        model = self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
                "streaming": True,
            }
        )
        assert model is not None

    def test_bind_tool_pydantic(
        self,
        model: BaseChatModel,
    ) -> None:
        if not self.has_tool_calling:
            return

        tools = [my_adder_tool, my_adder]

        for pydantic_model in TEST_PYDANTIC_MODELS:
            tools.extend([pydantic_model, pydantic_model.schema()])

        # Doing a mypy ignore here since some of the tools are from pydantic
        # BaseModel 2 which isn't typed properly yet. This will need to be fixed
        # so type checking does not become annoying to users.
        tool_model = model.bind_tools(tools, tool_choice="any")  # type: ignore
        assert isinstance(tool_model, RunnableBinding)

    @pytest.mark.parametrize("schema", TEST_PYDANTIC_MODELS)
    def test_with_structured_output(
        self,
        model: BaseChatModel,
        schema: Any,
    ) -> None:
        if not self.has_structured_output:
            return

        assert model.with_structured_output(schema) is not None

    def test_standard_params(self, model: BaseChatModel) -> None:
        from langchain_core.pydantic_v1 import BaseModel, ValidationError

        class ExpectedParams(BaseModel):
            ls_provider: str
            ls_model_name: str
            ls_model_type: Literal["chat"]
            ls_temperature: Optional[float]
            ls_max_tokens: Optional[int]
            ls_stop: Optional[List[str]]

        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)
        except ValidationError as e:
            pytest.fail(f"Validation error: {e}")

        # Test optional params
        model = self.chat_model_class(
            max_tokens=10, stop=["test"], **self.chat_model_params
        )
        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)
        except ValidationError as e:
            pytest.fail(f"Validation error: {e}")
