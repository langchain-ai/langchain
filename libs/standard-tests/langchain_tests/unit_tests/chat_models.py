"""Unit tests for chat models."""

import os
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
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

from langchain_tests.base import BaseStandardTests
from langchain_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION


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
        """The chat model class to test, e.g., `ChatParrotLink`."""
        ...

    @property
    def chat_model_params(self) -> dict:
        """Initialization parameters for the chat model.

        Example:

        .. code-block:: python

            @property
            def chat_model_params(self) -> dict:
                return {"model": "bird-brain-001", "temperature": 0}
        """
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
        """Boolean property indicating whether the chat model supports tool calling.

        By default, this is determined by whether the chat model's `bind_tools` method
        is overridden. It typically does not need to be overridden on the test class.
        """
        return self.chat_model_class.bind_tools is not BaseChatModel.bind_tools

    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice when used in tests.

        Some tests for tool calling features attempt to force tool calling via a
        `tool_choice` parameter. A common value for this parameter is "any". Defaults
        to `None`.

        Note: if the value is set to "tool_name", the name of the tool used in each
        test will be set as the value for `tool_choice`.

        Example:

        .. code-block:: python

            @property
            def tool_choice_value(self) -> Optional[str]:
                return "any"
        """
        return None

    @property
    def has_structured_output(self) -> bool:
        """Boolean property indicating whether the chat model supports structured
        output.

        By default, this is determined by whether the chat model's
        `with_structured_output` method is overridden. If the base implementation is
        intended to be used, this method should be overridden.

        See: https://python.langchain.com/docs/concepts/structured_outputs/

        Example:

        .. code-block:: python

            @property
            def has_structured_output(self) -> bool:
                return True
        """
        return (
            self.chat_model_class.with_structured_output
            is not BaseChatModel.with_structured_output
        )

    @property
    def supports_image_inputs(self) -> bool:
        """Boolean property indicating whether the chat model supports image inputs.
        Defaults to ``False``.

        If set to ``True``, the chat model will be tested using content blocks of the form

        .. code-block:: python

            [
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]

        See https://python.langchain.com/docs/concepts/multimodality/

        Example:

        .. code-block:: python

            @property
            def supports_image_inputs(self) -> bool:
                return True
        """
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """Boolean property indicating whether the chat model supports image inputs.
        Defaults to ``False``. No current tests are written for this feature."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Boolean property indicating whether the chat model returns usage metadata
        on invoke and streaming responses.

        ``usage_metadata`` is an optional dict attribute on AIMessages that track input
        and output tokens: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.UsageMetadata.html

        Example:

        .. code-block:: python

            @property
            def returns_usage_metadata(self) -> bool:
                return False
        """  # noqa: E501
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Boolean property indicating whether the chat model supports Anthropic-style
        inputs.

        These inputs might feature "tool use" and "tool result" content blocks, e.g.,

        .. code-block:: python

            [
                {"type": "text", "text": "Hmm let me think about that"},
                {
                    "type": "tool_use",
                    "input": {"fav_color": "green"},
                    "id": "foo",
                    "name": "color_picker",
                },
            ]

        If set to ``True``, the chat model will be tested using content blocks of this
        form.

        Example:

        .. code-block:: python

            @property
            def supports_anthropic_inputs(self) -> bool:
                return False
        """
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        """Boolean property indicating whether the chat model supports ToolMessages
        that include image content, e.g.,

        .. code-block:: python

            ToolMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
                tool_call_id="1",
                name="random_image",
            )

        If set to ``True``, the chat model will be tested with message sequences that
        include ToolMessages of this form.

        Example:

        .. code-block:: python

            @property
            def supports_image_tool_message(self) -> bool:
                return False
        """
        return False

    @property
    def supported_usage_metadata_details(
        self,
    ) -> Dict[
        Literal["invoke", "stream"],
        List[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        """Property controlling what usage metadata details are emitted in both invoke
        and stream.

        ``usage_metadata`` is an optional dict attribute on AIMessages that track input
        and output tokens: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.UsageMetadata.html

        It includes optional keys ``input_token_details`` and ``output_token_details``
        that can track usage details associated with special types of tokens, such as
        cached, audio, or reasoning.

        Only needs to be overridden if these details are supplied.
        """  # noqa: E501
        return {"invoke": [], "stream": []}


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
            ExpectedParams(**ls_params)  # type: ignore
        except ValidationErrorV1 as e:
            pytest.fail(f"Validation error: {e}")

        # Test optional params
        model = self.chat_model_class(
            max_tokens=10,
            stop=["test"],
            **self.chat_model_params,  # type: ignore
        )
        ls_params = model._get_ls_params()
        try:
            ExpectedParams(**ls_params)  # type: ignore
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
