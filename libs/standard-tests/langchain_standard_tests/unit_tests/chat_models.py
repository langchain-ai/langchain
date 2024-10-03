"""Unit tests for chat models."""

import os
from abc import abstractmethod
from typing import Any, List, Literal, Optional, Tuple, Type
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumpd, load
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool
from pydantic import BaseModel, Field, SecretStr
from pydantic.v1 import (
    BaseModel as BaseModelV1,
)
from pydantic.v1 import (
    Field as FieldV1,
)
from pydantic.v1 import (
    ValidationError as ValidationErrorV1,
)
from syrupy import SnapshotAssertion

from langchain_standard_tests.base import BaseStandardTests
from langchain_standard_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION


class Person(BaseModel):  # Used by some dependent tests. Should be deprecated.
    """Record attributes of a person."""

    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")


def generate_schema_pydantic_v1_from_2() -> Any:
    """Use to generate a schema from v1 namespace in pydantic 2."""
    if PYDANTIC_MAJOR_VERSION != 2:
        raise AssertionError("This function is only compatible with Pydantic v2.")

    class PersonB(BaseModelV1):
        """Record attributes of a person."""

        name: str = FieldV1(..., description="The name of the person.")
        age: int = FieldV1(..., description="The age of the person.")

    return PersonB


def generate_schema_pydantic() -> Any:
    """Works with either pydantic 1 or 2"""

    class PersonA(BaseModel):
        """Record attributes of a person."""

        name: str = Field(..., description="The name of the person.")
        age: int = Field(..., description="The age of the person.")

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


class ChatModelTests(BaseStandardTests):
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
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice when used in tests."""
        return None

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

    @property
    def supports_image_tool_message(self) -> bool:
        return False

    @property
    def supported_usage_metadata_details(
        self,
    ) -> List[
        Literal[
            "audio_input",
            "audio_output",
            "reasoning_output",
            "cache_read_input",
            "cache_creation_input",
        ]
    ]:
        return []


class ChatModelUnitTests(ChatModelTests):
    @property
    def standard_chat_model_params(self) -> dict:
        params = super().standard_chat_model_params
        params["api_key"] = "test"
        return params

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return {}, {}, {}

    def test_init(self) -> None:
        model = self.chat_model_class(
            **{**self.standard_chat_model_params, **self.chat_model_params}
        )
        assert model is not None

    def test_init_from_env(self) -> None:
        env_params, model_params, expected_attrs = self.init_from_env_params
        if env_params:
            with mock.patch.dict(os.environ, env_params):
                model = self.chat_model_class(**model_params)
            assert model is not None
            for k, expected in expected_attrs.items():
                actual = getattr(model, k)
                if isinstance(actual, SecretStr):
                    actual = actual.get_secret_value()
                assert actual == expected

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
            model_schema = (
                pydantic_model.model_json_schema()
                if hasattr(pydantic_model, "model_json_schema")
                else pydantic_model.schema()
            )
            tools.extend([pydantic_model, model_schema])

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
        class ExpectedParams(BaseModelV1):
            ls_provider: str
            ls_model_name: str
            ls_model_type: Literal["chat"]
            ls_temperature: Optional[float]
            ls_max_tokens: Optional[int]
            ls_stop: Optional[List[str]]

        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)
        except ValidationErrorV1 as e:
            pytest.fail(f"Validation error: {e}")

        # Test optional params
        model = self.chat_model_class(
            max_tokens=10, stop=["test"], **self.chat_model_params
        )
        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)
        except ValidationErrorV1 as e:
            pytest.fail(f"Validation error: {e}")

    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        if not self.chat_model_class.is_lc_serializable():
            return
        env_params, model_params, expected_attrs = self.init_from_env_params
        with mock.patch.dict(os.environ, env_params):
            ser = dumpd(model)
            assert ser == snapshot(name="serialized")
            assert model.dict() == load(dumpd(model)).dict()
