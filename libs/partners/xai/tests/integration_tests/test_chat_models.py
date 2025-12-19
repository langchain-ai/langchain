"""Integration tests for ChatXAI specific features."""

from __future__ import annotations

from typing import Literal

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk

from langchain_xai import ChatXAI

MODEL_NAME = "grok-4-fast-reasoning"


@pytest.mark.parametrize("output_version", ["", "v1"])
def test_reasoning(output_version: Literal["", "v1"]) -> None:
    """Test reasoning features.

    !!! note
        `grok-4` does not return `reasoning_content`, but may optionally return
        encrypted reasoning content if `use_encrypted_content` is set to `True`.
    """
    # Test reasoning effort
    if output_version:
        chat_model = ChatXAI(
            model="grok-3-mini",
            reasoning_effort="low",
            output_version=output_version,
        )
    else:
        chat_model = ChatXAI(
            model="grok-3-mini",
            reasoning_effort="low",
        )
    input_message = "What is 3^3?"
    response = chat_model.invoke(input_message)
    assert response.content
    assert response.additional_kwargs["reasoning_content"]

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in chat_model.stream(input_message):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]

    # Check that we can access reasoning content blocks
    assert response.content_blocks
    reasoning_content = (
        block for block in response.content_blocks if block["type"] == "reasoning"
    )
    assert len(list(reasoning_content)) >= 1

    # Test that passing message with reasoning back in works
    follow_up_message = "Based on your reasoning, what is 4^4?"
    followup = chat_model.invoke([input_message, response, follow_up_message])
    assert followup.content
    assert followup.additional_kwargs["reasoning_content"]
    followup_reasoning = (
        block for block in followup.content_blocks if block["type"] == "reasoning"
    )
    assert len(list(followup_reasoning)) >= 1

    # Test passing in a ReasoningContentBlock
    response_metadata = {"model_provider": "xai"}
    if output_version:
        response_metadata["output_version"] = output_version
    msg_w_reasoning = AIMessage(
        content_blocks=response.content_blocks,
        response_metadata=response_metadata,
    )
    followup_2 = chat_model.invoke(
        [msg_w_reasoning, "Based on your reasoning, what is 5^5?"]
    )
    assert followup_2.content
    assert followup_2.additional_kwargs["reasoning_content"]


def test_web_search() -> None:
    llm = ChatXAI(
        model=MODEL_NAME,
        search_parameters={"mode": "on", "max_search_results": 3},
    )

    # Test invoke
    response = llm.invoke("Provide me a digest of world news in the last 24 hours.")
    assert response.content
    assert response.additional_kwargs["citations"]
    assert len(response.additional_kwargs["citations"]) <= 3

    # Test streaming
    full = None
    for chunk in llm.stream("Provide me a digest of world news in the last 24 hours."):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["citations"]
    assert len(full.additional_kwargs["citations"]) <= 3
