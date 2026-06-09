"""Standard LangChain interface tests."""

from typing import Any

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

    @property
    def has_tool_calling(self) -> bool:
        # The sonar family does not support client-side function tools: the API
        # returns 400 "Tool calling is not supported for this model". Tool
        # calling is exercised by TestPerplexityResponsesStandard below, which
        # runs against the Responses (Agent) API.
        return False

    @pytest.mark.xfail(reason="TODO: handle in integration.")
    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        super().test_double_messages_conversation(model)

    @pytest.mark.xfail(reason="Raises 400: Custom stop words not supported.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)

    # TODO, API regressed for some reason after 2025-04-15


class TestPerplexityResponsesStandard(ChatModelIntegrationTests):
    """Standard tests on the Responses (Agent) API, which supports tool calling.

    Client-side function tools require the Responses route and a tool-capable
    model (the `sonar` family does not support them), so the tool-calling test
    family runs here rather than on `TestPerplexityStandard`.
    """

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatPerplexity

    @property
    def chat_model_params(self) -> dict:
        return {"model": "openai/gpt-5.5", "use_responses_api": True}

    @property
    def has_tool_choice(self) -> bool:
        # The Perplexity Responses (Agent) API does not support `tool_choice`
        # (`_to_responses_payload` raises `ValueError`), so forced tool
        # selection cannot be used here. The model still calls tools when the
        # prompt warrants it, which is what the tool tests assert.
        return False

    # These two tests hard-code `tool_choice="any"` (they are not gated by
    # `has_tool_choice`) to force a tool call, which the Responses route rejects
    # with `ValueError`. They are xfailed here rather than disabling the whole
    # tool-calling family; every other tool test runs and passes.
    @pytest.mark.xfail(
        reason="Responses (Agent) API does not support tool_choice; this test "
        "hard-codes tool_choice='any'."
    )
    def test_unicode_tool_call_integration(self, *args: Any, **kwargs: Any) -> None:
        super().test_unicode_tool_call_integration(*args, **kwargs)

    @pytest.mark.xfail(
        reason="Responses (Agent) API does not support tool_choice; this test "
        "hard-codes tool_choice='any'."
    )
    def test_structured_few_shot_examples(self, *args: Any, **kwargs: Any) -> None:
        super().test_structured_few_shot_examples(*args, **kwargs)
