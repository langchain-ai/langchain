"""Test chat model integration using standard integration tests."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_ollama.chat_models import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"


class TestChatOllama(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": DEFAULT_MODEL_NAME}

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        # TODO: update after Ollama implements
        # https://github.com/ollama/ollama/blob/main/docs/openai.md#supported-request-fields
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason=(
            "Will sometime encounter AssertionErrors where tool responses are "
            "`'3'` instead of `3`"
        )
    )
    def test_tool_calling(self, model: BaseChatModel) -> None:
        super().test_tool_calling(model)

    @pytest.mark.xfail(
        reason=(
            "Will sometime encounter AssertionErrors where tool responses are "
            "`'3'` instead of `3`"
        )
    )
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        await super().test_tool_calling_async(model)

    @pytest.mark.xfail(
        reason=(
            "Will sometimes fail due to Ollama's inconsistent tool call argument "
            "structure (see https://github.com/ollama/ollama/issues/6155). "
            "Args may contain unexpected keys like 'conversations' instead of "
            "empty dict."
        )
    )
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)
