"""Standard LangChain interface tests for `_ChatOpenAICodex`.

Drives the full `ChatModelIntegrationTests` suite against the Codex
backend. The module-level `pytestmark = pytest.mark.vcr` makes every
inherited test record/replay through VCR so cassettes recorded once
locally replay without a live OAuth token in CI.

Capability flags below reflect what the Codex ChatGPT-subscription
endpoint currently exposes (image inputs, PDF inputs, audio inputs,
JSON-mode `response_format`, Anthropic-format inputs all work). The
divergence from `ChatOpenAI`'s defaults is intentional — Codex is a
subset surface, so a few `ChatModelIntegrationTests` cases are xfailed
with documented reasons.
"""

from __future__ import annotations

import os
from typing import Any, Literal, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai.chat_models.codex import _ChatOpenAICodex

pytestmark = pytest.mark.vcr

MODEL_NAME = os.getenv("CODEX_MODEL", "gpt-5.5")
TERSE_INSTRUCTIONS = "You are terse. Answer in five words or fewer."


class TestChatOpenAICodexStandard(ChatModelIntegrationTests):
    """Standard chat-model integration suite, configured for Codex.

    Capability properties below override the upstream defaults to match
    what the Codex backend supports (or fails on) — flip any to `False`
    if Codex regresses on the corresponding capability, then re-record
    cassettes via the recording script.
    """

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return _ChatOpenAICodex

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME, "instructions": TERSE_INSTRUCTIONS}

    @property
    def model_override_value(self) -> str | None:
        # The Codex ChatGPT-subscription backend exposes a single model
        # to this client; reuse `MODEL_NAME` so the override path exercises
        # the per-call `model=` plumbing without depending on a second
        # account-eligible model.
        return MODEL_NAME

    # -- Capability flags -------------------------------------------------
    # All currently confirmed working against the ChatGPT-subscription
    # Codex backend at recording time. Flip a flag to `False` and
    # re-record if Codex stops accepting a capability.

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_pdf_tool_message(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True

    @property
    def enable_vcr_tests(self) -> bool:
        return True

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        # The Codex backend reports `reasoning_output` for reasoning-enabled
        # models; cache and audio metadata are not surfaced through the
        # subscription endpoint.
        return {"invoke": ["reasoning_output"], "stream": ["reasoning_output"]}

    # -- Helpers used by the shared suite ---------------------------------

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        llm = _ChatOpenAICodex(
            model=MODEL_NAME,
            instructions=TERSE_INSTRUCTIONS,
            reasoning={"effort": "medium", "summary": "auto"},
        )
        prompt = "What was the 3rd highest building in 2000?"
        return _invoke(llm, prompt, stream)

    # -- Codex-specific xfails --------------------------------------------

    @pytest.mark.xfail(reason="Codex backend does not honor `stop` sequences.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)

    @pytest.mark.xfail(
        reason=(
            "Few-shot helper generates a fresh tool-call UUID per invocation, "
            "so the recorded request body never matches on replay. Tracked "
            "separately; this test needs a deterministic call_id or a "
            "matcher exception in conftest before it can be cassette-backed."
        )
    )
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: Any
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool)


def _invoke(llm: _ChatOpenAICodex, prompt: str, stream: bool) -> AIMessage:
    if stream:
        full = None
        for chunk in llm.stream(prompt):
            full = full + chunk if full else chunk  # type: ignore[operator]
        return cast(AIMessage, full)
    return cast(AIMessage, llm.invoke(prompt))
