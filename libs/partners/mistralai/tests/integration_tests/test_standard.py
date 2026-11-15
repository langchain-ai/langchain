"""Standard LangChain interface tests."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_mistralai import ChatMistralAI
from tests.integration_tests._rate_limiter import rate_limiter


class TestMistralStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatMistralAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "mistral-large-latest",
            "rate_limiter": rate_limiter,
        }

    @property
    def has_tool_choice(self) -> bool:
        # mistral-large-latest hangs when tool_choice is specified
        return False

    @property
    def structured_output_kwargs(self) -> dict:
        # TODO: make json_schema the default `with_structured_output` method
        return {"method": "json_schema"}

    @property
    def supports_json_mode(self) -> bool:
        return True

    @pytest.mark.xfail(reason=("MistralAI inconsistently fails to return valid fields"))
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)
