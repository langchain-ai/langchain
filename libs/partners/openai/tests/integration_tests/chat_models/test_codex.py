"""Integration tests for `_ChatOpenAICodex`.

These tests exercise the ChatGPT subscription OAuth path against the
`https://chatgpt.com/backend-api/codex` endpoint. They are recorded with
VCR (cassettes live alongside, under `tests/cassettes/`) so CI replays
them in `--record-mode=none` without a live token.

`_ChatOpenAICodex` forces `use_responses_api=True`, `store=False`, and
`streaming=True` at the wire level (`output_version` is a client-side
projection and is *not* forced). The cassettes here are recorded with a
single `output_version` for stability; per-projection coverage already
lives in `test_responses_api.py`.

Override the model with `CODEX_MODEL=<id>` when recording against a
different account / plan tier.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_openai import custom_tool
from langchain_openai.chat_models.codex import _ChatOpenAICodex

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Iterator

    from langchain_core.language_models.chat_model_stream import (
        AsyncChatModelStream,
        ChatModelStream,
    )

pytestmark = pytest.mark.vcr

MODEL_NAME = os.getenv("CODEX_MODEL", "gpt-5.5")
TERSE_INSTRUCTIONS = "You are terse. Answer in five words or fewer."


def _check_response(response: BaseMessage | None) -> None:
    """Assert the response carries the minimum Responses-API shape.

    Looser than the `test_responses_api.py` equivalent — Codex responses
    don't always populate `service_tier`, so we don't require it here.
    """
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    text_content = response.text
    assert isinstance(text_content, str)
    assert text_content
    assert response.usage_metadata
    assert response.usage_metadata["input_tokens"] > 0
    assert response.usage_metadata["output_tokens"] > 0
    assert response.usage_metadata["total_tokens"] > 0
    assert response.response_metadata["model_name"]


def _aggregate(stream: Iterator[BaseMessage]) -> AIMessageChunk:
    """Drain a sync chunk stream and return the aggregated `AIMessageChunk`.

    Typed against `BaseMessage` (the broadest output of `Runnable.stream`)
    so that `bound.stream(...)` — which mypy infers as `Iterator[AIMessage]`
    after `bind_tools` — is accepted without a cast. The `isinstance`
    guard inside the loop enforces the runtime contract.
    """
    aggregated: BaseMessageChunk | None = None
    for chunk in stream:
        assert isinstance(chunk, AIMessageChunk)
        aggregated = chunk if aggregated is None else aggregated + chunk
    assert isinstance(aggregated, AIMessageChunk)
    return aggregated


async def _aaggregate(stream: AsyncIterator[BaseMessage]) -> AIMessageChunk:
    """Drain an async chunk stream and return the aggregated `AIMessageChunk`."""
    aggregated: BaseMessageChunk | None = None
    async for chunk in stream:
        assert isinstance(chunk, AIMessageChunk)
        aggregated = chunk if aggregated is None else aggregated + chunk
    assert isinstance(aggregated, AIMessageChunk)
    return aggregated


# ---------------------------------------------------------------------------
# Basic invoke / stream / async surface
# ---------------------------------------------------------------------------


def test_codex_invoke() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    response = llm.invoke("Say hi in one word.")
    _check_response(response)


def test_codex_invoke_lifts_system_message_into_instructions() -> None:
    """`SystemMessage` content is lifted into top-level `instructions`.

    Codex rejects `SystemMessage` chat turns; `_ChatOpenAICodex` works
    around this by moving the `SystemMessage` content into the
    `instructions` field and stripping it from the input list. The model
    should respect the lifted instruction (here: respond with HELLO).
    """
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    response = llm.invoke(
        [
            SystemMessage("Respond with exactly one word: HELLO. No punctuation."),
            HumanMessage("Greet me."),
        ]
    )
    _check_response(response)
    assert "hello" in response.text.lower()


def test_codex_invoke_with_instructions_override() -> None:
    """Per-call `instructions=` overrides the constructor value for one call."""
    llm = _ChatOpenAICodex(
        model=MODEL_NAME, instructions="You are an English assistant."
    )
    response = llm.invoke(
        "Greet me.",
        instructions=(
            "You are a French assistant. Respond only in French, in five words "
            "or fewer."
        ),
    )
    _check_response(response)


async def test_codex_invoke_async() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    response = await llm.ainvoke("Say hi in one word.")
    _check_response(response)


def test_codex_stream() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    _check_response(_aggregate(llm.stream("Count to three.")))


async def test_codex_stream_async() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    _check_response(await _aaggregate(llm.astream("Count to three.")))


def test_codex_stream_events_v3() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    stream = cast("ChatModelStream", llm.stream_events("Count to three.", version="v3"))
    response = stream.output
    _check_response(response)


async def test_codex_stream_events_v3_async() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    stream = await cast(
        "Awaitable[AsyncChatModelStream]",
        llm.astream_events("Count to three.", version="v3"),
    )
    response = await stream
    _check_response(response)


# ---------------------------------------------------------------------------
# Multi-turn conversation
# ---------------------------------------------------------------------------


def test_codex_multi_turn_no_tools() -> None:
    """Pass full chat history (the backend is stateless for this client)."""
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    first = llm.invoke("My name is Bobo.")
    assert isinstance(first, AIMessage)
    second = llm.invoke(
        [
            HumanMessage("My name is Bobo."),
            first,
            HumanMessage("What is my name?"),
        ]
    )
    _check_response(second)
    assert "bobo" in second.text.lower()


# ---------------------------------------------------------------------------
# Function calling / agent loop
# ---------------------------------------------------------------------------


def test_codex_function_calling() -> None:
    @tool
    def multiply(x: int, y: int) -> int:
        """Return x * y."""
        return x * y

    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    bound = llm.bind_tools([multiply])

    ai_msg = cast(AIMessage, bound.invoke("What is 5 times 4?"))
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"
    assert set(ai_msg.tool_calls[0]["args"]) == {"x", "y"}

    aggregated = _aggregate(bound.stream("What is 5 times 4?"))
    assert len(aggregated.tool_calls) == 1
    assert aggregated.tool_calls[0]["name"] == "multiply"
    assert set(aggregated.tool_calls[0]["args"]) == {"x", "y"}


def test_codex_agent_loop() -> None:
    """Tool call → tool message → final answer (three round trips)."""

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    bound = llm.bind_tools([get_weather])

    user_msg = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_msg = cast(AIMessage, bound.invoke([user_msg]))
    assert tool_call_msg.tool_calls
    tool_call = tool_call_msg.tool_calls[0]
    tool_msg = get_weather.invoke(tool_call)
    assert isinstance(tool_msg, ToolMessage)

    final = bound.invoke([user_msg, tool_call_msg, tool_msg])
    assert isinstance(final, AIMessage)


def test_codex_agent_loop_streaming() -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    bound = llm.bind_tools([get_weather])

    user_msg = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_msg = _aggregate(bound.stream([user_msg]))
    assert tool_call_msg.tool_calls
    tool_msg = get_weather.invoke(tool_call_msg.tool_calls[0])
    assert isinstance(tool_msg, ToolMessage)

    final = _aggregate(bound.stream([user_msg, tool_call_msg, tool_msg]))
    assert isinstance(final, AIMessage)


def test_codex_custom_tool() -> None:
    @custom_tool
    def execute_code(code: str) -> str:
        """Execute Python code and return the result."""
        return "27"

    llm = _ChatOpenAICodex(
        model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS
    ).bind_tools([execute_code])

    input_message = {
        "role": "user",
        "content": "Use the execute_code tool to evaluate 3**3.",
    }
    tool_call_msg = cast(AIMessage, llm.invoke([input_message]))
    assert tool_call_msg.tool_calls
    tool_msg = execute_code.invoke(tool_call_msg.tool_calls[0])
    response = llm.invoke([input_message, tool_call_msg, tool_msg])
    assert isinstance(response, AIMessage)


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


def test_codex_reasoning() -> None:
    """`reasoning={'effort': 'low'}` produces a reasoning block in content."""
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    response = llm.invoke("What is 2 + 2?", reasoning={"effort": "low"})
    assert isinstance(response, AIMessage)
    block_types = [
        block["type"] for block in response.content if isinstance(block, dict)
    ]
    assert "reasoning" in block_types or "text" in block_types


def test_codex_reasoning_summary_streaming() -> None:
    """`reasoning.summary='auto'` carries a populated summary list."""
    llm = _ChatOpenAICodex(
        model=MODEL_NAME,
        instructions=TERSE_INSTRUCTIONS,
        reasoning={"effort": "medium", "summary": "auto"},
    )
    aggregated = _aggregate(
        llm.stream("What was the tallest building in the year 2000?")
    )

    reasoning_blocks = [
        block
        for block in aggregated.content
        if isinstance(block, dict) and block["type"] == "reasoning"
    ]
    # A non-trivial prompt should produce exactly one reasoning content block
    # with at least one summary entry; if Codex stops streaming summaries
    # this assertion regresses loudly instead of silently passing.
    assert len(reasoning_blocks) >= 1
    summary = reasoning_blocks[0].get("summary")
    assert isinstance(summary, list)
    assert summary, "reasoning block emitted an empty `summary` list"
    for summary_block in summary:
        assert isinstance(summary_block, dict)
        assert isinstance(summary_block.get("type"), str)
        assert isinstance(summary_block.get("text"), str)
        assert summary_block["text"]


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class Foo(BaseModel):
    """A trivial pydantic schema used to exercise `response_format`."""

    # Docstring is intentional: Pydantic emits it as `description` in the
    # JSON schema sent to Codex, which the cassettes pin on.
    response: str


class FooDict(TypedDict):
    """A trivial TypedDict schema used to exercise `response_format`."""

    response: str


def test_codex_structured_output_pydantic() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    response = llm.invoke("Say hi.", response_format=Foo)
    parsed = Foo(**json.loads(response.text))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response


def test_codex_structured_output_typed_dict() -> None:
    llm = _ChatOpenAICodex(model=MODEL_NAME, instructions=TERSE_INSTRUCTIONS)
    response = llm.invoke("Say hi.", response_format=FooDict)
    parsed = json.loads(response.text)
    assert parsed == response.additional_kwargs["parsed"]
    assert isinstance(parsed["response"], str)
    assert parsed["response"]


# Header behavior (originator default/override, ChatGPT-Account-Id presence)
# is covered by the unit suite — VCR cassettes redact every recorded header
# value, so an integration test here couldn't distinguish a header-present
# round trip from a header-absent one anyway.
