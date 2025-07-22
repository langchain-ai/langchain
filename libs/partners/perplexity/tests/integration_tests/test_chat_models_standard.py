"""Standard LangChain interface tests."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_perplexity import ChatPerplexity


class TestPerplexityStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatPerplexity

    @property
    def chat_model_params(self) -> dict:
        return {"model": "sonar"}

    @pytest.mark.xfail(reason="TODO: handle in integration.")
    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        super().test_double_messages_conversation(model)

    @pytest.mark.xfail(reason="Raises 400: Custom stop words not supported.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)

    @pytest.mark.xfail(reason="TODO, this regressed for some reason after 2025-04-15")
    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    def test_structured_output(self, model: BaseChatModel, schema_type: str) -> None:
        super().test_structured_output(model, schema_type)

    @pytest.mark.xfail(reason="TODO, this regressed for some reason after 2025-04-15")
    @pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
    async def test_structured_output_async(
        self, model: BaseChatModel, schema_type: str
    ) -> None:
        await super().test_structured_output_async(model, schema_type)

    @pytest.mark.xfail(reason="TODO, this regressed for some reason after 2025-04-15")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="TODO, this regressed for some reason after 2025-04-15")
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)
