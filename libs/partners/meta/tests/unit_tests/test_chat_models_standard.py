"""Standard LangChain interface tests"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,
)
from langchain_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1

from langchain_meta import ChatLlama


def generate_schema_pydantic_v1_from_2() -> Any:
    """
    Use to generate a schema from v1 namespace in pydantic 2.

    :private:
    """
    if PYDANTIC_MAJOR_VERSION != 2:
        raise AssertionError("This function is only compatible with Pydantic v2.")

    class PersonB(BaseModelV1):
        """Record attributes of a person."""

        name: str = FieldV1(..., description="The name of the person.")
        age: int = FieldV1(..., description="The age of the person.")

    return PersonB


def generate_schema_pydantic() -> Any:
    """
    Works with either pydantic 1 or 2

    :private:
    """

    class PersonA(BaseModel):
        """Record attributes of a person."""

        name: str = Field(..., description="The name of the person.")
        age: int = Field(..., description="The age of the person.")

    return PersonA


TEST_PYDANTIC_MODELS = [generate_schema_pydantic()]

if PYDANTIC_MAJOR_VERSION == 2:
    TEST_PYDANTIC_MODELS.append(generate_schema_pydantic_v1_from_2())


class TestLlamaStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatLlama

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Llama-3.3-8B-Instruct"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "LLAMA_API_KEY": "api_key",
            },
            {
                "model": "Llama-3.3-8B-Instruct",
            },
            {
                "llama_api_key": "api_key",
                "llama_api_base": "https://api.llama.com/compat/v1/",
            },
        )

    @pytest.mark.parametrize("schema", TEST_PYDANTIC_MODELS)
    @pytest.mark.xfail(
        reason="Llama models only support 'json_schema' method,"
        "other methods are expected to fail."
    )
    def test_with_structured_output(
        self,
        model: BaseChatModel,
        schema: Any,
    ) -> None:
        """
        Test that Llama models only support
        the 'json_schema' method for structured output.
        """
        if not self.has_structured_output:
            return

        # Test default method works (should be json_schema)
        assert model.with_structured_output(schema) is not None

        # Test json_schema method explicitly
        for strict in [None, False, True]:
            assert model.with_structured_output(
                schema, method="json_schema", strict=strict
            )

        # Test that other methods raise ValueError
        for method in ["function_calling", "json_mode"]:
            with pytest.raises(
                ValueError, match="Only 'json_schema' method is supported"
            ):
                model.with_structured_output(schema, method=method)
