"""Standard LangChain interface tests."""

from typing import Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class TestHuggingFaceEndpoint(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatHuggingFace

    @property
    def chat_model_params(self) -> dict:
        llm = HuggingFaceEndpoint(  # type: ignore[call-arg]
            repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            task="conversational",
            provider="fireworks-ai",
            temperature=0,
        )
        return {"llm": llm}

    @pytest.fixture
    def model(self) -> BaseChatModel:
        return self.chat_model_class(**self.chat_model_params)  # type: ignore[call-arg]

    @pytest.mark.xfail(
        reason=("Overrding, testing only typed dict and json schema structured output")
    )
    @pytest.mark.parametrize("schema_type", ["typeddict", "json_schema"])
    def test_structured_output(
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> None:
        super().test_structured_output(model, schema_type)

    @pytest.mark.xfail(
        reason=("Overrding, testing only typed dict and json schema structured output")
    )
    @pytest.mark.parametrize("schema_type", ["typeddict", "json_schema"])
    async def test_structured_output_async(
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> None:
        super().test_structured_output(model, schema_type)

    @pytest.mark.xfail(reason=("Pydantic structured output is not supported"))
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason=("Pydantic structured output is not supported"))
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(
            model, my_adder_tool=my_adder_tool
        )

    @property
    def has_tool_choice(self) -> bool:
        return False
