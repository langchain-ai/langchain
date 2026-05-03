"""Tests for Anthropic tool-call ID sanitization middleware."""

import logging
import re
import warnings
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.runtime import Runtime

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_anthropic.middleware import AnthropicToolIdSanitizationMiddleware

_ANTHROPIC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class _FakeModel(BaseChatModel):
    """Stand-in non-Anthropic model for unsupported-model tests."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok", id="0"))]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok", id="0"))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake"


def _make_request(
    messages: list[BaseMessage], *, anthropic: bool = True
) -> ModelRequest:
    """Build a `ModelRequest` for testing."""
    if anthropic:
        mock_model = MagicMock(spec=ChatAnthropic)
        mock_model._llm_type = "anthropic-chat"
        model: BaseChatModel = cast(BaseChatModel, mock_model)
    else:
        model = _FakeModel()
    return ModelRequest(
        model=model,
        messages=cast(list[AnyMessage], messages),
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=cast(AgentState[Any], {"messages": messages}),
        runtime=cast(Runtime, object()),
        model_settings={},
    )


def _capture(
    messages: list[BaseMessage],
) -> tuple[
    AnthropicToolIdSanitizationMiddleware,
    ModelRequest,
    list[ModelRequest],
]:
    """Wire a middleware + request + capturing handler list for a test."""
    middleware = AnthropicToolIdSanitizationMiddleware()
    request = _make_request(messages)
    captured: list[ModelRequest] = []

    return middleware, request, captured


def _handler_factory(captured: list[ModelRequest]) -> Any:
    def _handler(req: ModelRequest) -> ModelResponse:
        captured.append(req)
        return ModelResponse(result=[AIMessage(content="ok")])

    return _handler


def _valid_tool_call_id(value: str | None) -> str:
    """Assert that an optional tool-call ID is present and Anthropic-safe."""
    assert value is not None
    assert _ANTHROPIC_ID_RE.match(value)
    return value


def test_passthrough_when_all_ids_valid() -> None:
    """Valid IDs short-circuit: every message reaches the handler unchanged."""
    messages: list[BaseMessage] = [
        HumanMessage("hi"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "grep", "args": {}, "id": "toolu_01abc", "type": "tool_call"}
            ],
        ),
        ToolMessage(content="done", tool_call_id="toolu_01abc"),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    assert len(captured) == 1
    # Per-message identity: each individual message is the original instance.
    for original, sent in zip(request.messages, captured[0].messages, strict=True):
        assert sent is original


def test_rewrites_kimi_k2_style_ids_in_pairs() -> None:
    """Bad IDs on AIMessage tool_calls and matching ToolMessage are rewritten."""
    messages: list[BaseMessage] = [
        HumanMessage("hi"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "grep",
                    "args": {"q": "x"},
                    "id": "functions.grep:0",
                    "type": "tool_call",
                },
                {
                    "name": "grep",
                    "args": {"q": "y"},
                    "id": "functions.grep:1",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="r0", tool_call_id="functions.grep:0"),
        ToolMessage(content="r1", tool_call_id="functions.grep:1"),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = sent[1]
    assert isinstance(ai, AIMessage)
    new_ids = [_valid_tool_call_id(tc["id"]) for tc in ai.tool_calls]
    tool_msgs = [m for m in sent if isinstance(m, ToolMessage)]
    assert [tm.tool_call_id for tm in tool_msgs] == new_ids
    assert new_ids[0] != new_ids[1]


def test_rewrite_is_deterministic_across_invocations() -> None:
    """The same input produces the same sanitized IDs across separate calls."""
    payload: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "grep",
                    "args": {},
                    "id": "functions.grep:0",
                    "type": "tool_call",
                },
                {
                    "name": "grep",
                    "args": {},
                    "id": "functions.grep:1",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="r0", tool_call_id="functions.grep:0"),
        ToolMessage(content="r1", tool_call_id="functions.grep:1"),
    ]

    def _run() -> list[str]:
        # Build a fresh copy each invocation so the two runs are independent.
        msgs = [m.model_copy(deep=True) for m in payload]
        middleware = AnthropicToolIdSanitizationMiddleware()
        request = _make_request(msgs)
        captured: list[ModelRequest] = []
        middleware.wrap_model_call(request, _handler_factory(captured))
        ai = cast(AIMessage, captured[0].messages[0])
        return [_valid_tool_call_id(tc["id"]) for tc in ai.tool_calls]

    assert _run() == _run()


def test_rewrites_anthropic_content_blocks() -> None:
    """`tool_use` and `tool_result` content blocks have their IDs rewritten too."""
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {"type": "text", "text": "calling"},
                {
                    "type": "tool_use",
                    "id": "functions.read_file:0",
                    "name": "read_file",
                    "input": {},
                },
            ],
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {},
                    "id": "functions.read_file:0",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "functions.read_file:0",
                    "content": "hi",
                },
            ],
            tool_call_id="functions.read_file:0",
        ),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = sent[0]
    tm = sent[1]
    assert isinstance(ai, AIMessage)
    assert isinstance(tm, ToolMessage)

    block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    tool_call_id = _valid_tool_call_id(ai.tool_calls[0]["id"])
    result_id = next(
        b["tool_use_id"]
        for b in tm.content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )

    assert block_id == tool_call_id == tm.tool_call_id == result_id
    for value in (block_id, tool_call_id, tm.tool_call_id, result_id):
        assert _ANTHROPIC_ID_RE.match(value)


def test_drift_between_tool_calls_and_tool_use_blocks_is_corrected() -> None:
    """When `tool_calls[i].id` and the i-th tool_use block disagree, alignment wins.

    Without correction, two distinct invalid IDs would map to two different
    safe IDs (`a_b` vs `a_b_<digest>`), and Anthropic would 400 on the
    mismatched pair. The middleware forces both to share `tool_calls[i].id`.
    """
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "a:b",  # drift — different illegal id
                    "name": "tool",
                    "input": {},
                },
            ],
            tool_calls=[
                {
                    "name": "tool",
                    "args": {},
                    "id": "a.b",  # canonical
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "a:b",
                    "content": "r",
                },
            ],
            tool_call_id="a:b",
        ),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = cast(AIMessage, sent[0])
    tm = cast(ToolMessage, sent[1])

    final_tc_id = _valid_tool_call_id(ai.tool_calls[0]["id"])
    final_block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    final_result_id = next(
        b["tool_use_id"]
        for b in tm.content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )

    assert final_tc_id == final_block_id == tm.tool_call_id == final_result_id


def test_valid_drift_between_tool_calls_and_tool_use_blocks_is_corrected() -> None:
    """Valid but mismatched tool-call and `tool_use` IDs are aligned."""
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_content",
                    "name": "tool",
                    "input": {},
                },
            ],
            tool_calls=[
                {
                    "name": "tool",
                    "args": {},
                    "id": "toolu_call",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_content",
                    "content": "r",
                },
            ],
            tool_call_id="toolu_content",
        ),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = cast(AIMessage, sent[0])
    tm = cast(ToolMessage, sent[1])
    block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    result_id = next(
        b["tool_use_id"]
        for b in tm.content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )

    assert ai.tool_calls[0]["id"] == "toolu_call"
    assert block_id == "toolu_call"
    assert tm.tool_call_id == "toolu_call"
    assert result_id == "toolu_call"


def test_two_invalid_ids_with_same_base_get_distinct_safe_ids() -> None:
    """`a.b` and `a:b` both sanitize to base `a_b` — second falls back to suffix.

    Exercises the sha256-suffix branch of `_make_safe_id`. Both AIMessages
    use the same callable name so name-based heuristics can't disambiguate;
    only the `_make_safe_id` collision logic produces distinct safe IDs.
    """
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[{"name": "tool", "args": {}, "id": "a.b", "type": "tool_call"}],
        ),
        ToolMessage(content="r1", tool_call_id="a.b"),
        AIMessage(
            content="",
            tool_calls=[{"name": "tool", "args": {}, "id": "a:b", "type": "tool_call"}],
        ),
        ToolMessage(content="r2", tool_call_id="a:b"),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    first_id = _valid_tool_call_id(cast(AIMessage, sent[0]).tool_calls[0]["id"])
    second_id = _valid_tool_call_id(cast(AIMessage, sent[2]).tool_calls[0]["id"])

    # Both sanitize cleanly, both are valid, both are distinct.
    assert first_id != second_id
    # The first invalid wins the base; the second gets the suffix.
    assert first_id == "a_b"
    assert second_id.startswith("a_b_")
    # Pairs survive: ToolMessage ids match their AIMessage counterparts.
    assert cast(ToolMessage, sent[1]).tool_call_id == first_id
    assert cast(ToolMessage, sent[3]).tool_call_id == second_id


def test_state_is_not_mutated() -> None:
    """Original messages on the request and in graph state are not mutated."""
    original_id = "functions.grep:0"
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "grep", "args": {}, "id": original_id, "type": "tool_call"}
            ],
        ),
        ToolMessage(content="r", tool_call_id=original_id),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    assert request.messages[0].tool_calls[0]["id"] == original_id  # type: ignore[union-attr]
    assert cast(ToolMessage, request.messages[1]).tool_call_id == original_id
    assert captured[0].messages is not request.messages


def test_collision_avoidance_with_existing_valid_id() -> None:
    """A bad ID that sanitizes to an already-used valid ID gets a hash suffix."""
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "grep",
                    "args": {},
                    "id": "functions_grep_0",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="a", tool_call_id="functions_grep_0"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "grep",
                    "args": {},
                    "id": "functions.grep:0",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="b", tool_call_id="functions.grep:0"),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    first_id = _valid_tool_call_id(cast(AIMessage, sent[0]).tool_calls[0]["id"])
    second_id = _valid_tool_call_id(cast(AIMessage, sent[2]).tool_calls[0]["id"])

    assert first_id == "functions_grep_0"
    assert second_id != first_id


def test_mixed_valid_and_invalid_ids_in_same_message() -> None:
    """Valid IDs are preserved while siblings with invalid IDs are rewritten."""
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "good",
                    "args": {},
                    "id": "toolu_clean",
                    "type": "tool_call",
                },
                {
                    "name": "bad",
                    "args": {},
                    "id": "functions.bad:0",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="g", tool_call_id="toolu_clean"),
        ToolMessage(content="b", tool_call_id="functions.bad:0"),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = cast(AIMessage, sent[0])
    valid_id = _valid_tool_call_id(ai.tool_calls[0]["id"])
    rewritten_id = _valid_tool_call_id(ai.tool_calls[1]["id"])

    assert valid_id == "toolu_clean"  # untouched
    assert rewritten_id != "functions.bad:0"
    assert cast(ToolMessage, sent[1]).tool_call_id == valid_id
    assert cast(ToolMessage, sent[2]).tool_call_id == rewritten_id


def test_rewrites_server_and_mcp_block_types() -> None:
    """`server_tool_use` / `mcp_tool_use` and their result variants are also handled."""
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {
                    "type": "server_tool_use",
                    "id": "srv.1",
                    "name": "web_search",
                    "input": {},
                },
                {
                    "type": "mcp_tool_use",
                    "id": "mcp.2",
                    "name": "fetch",
                    "input": {},
                },
            ],
            tool_calls=[],
        ),
        ToolMessage(
            content=[
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv.1",
                    "content": "hits",
                },
            ],
            tool_call_id="srv.1",
        ),
        ToolMessage(
            content=[
                {
                    "type": "mcp_tool_result",
                    "tool_use_id": "mcp.2",
                    "content": "data",
                },
            ],
            tool_call_id="mcp.2",
        ),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = cast(AIMessage, sent[0])
    server_block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "server_tool_use"
    )
    mcp_block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "mcp_tool_use"
    )

    for value in (
        server_block_id,
        mcp_block_id,
        cast(ToolMessage, sent[1]).tool_call_id,
        cast(ToolMessage, sent[2]).tool_call_id,
    ):
        assert _ANTHROPIC_ID_RE.match(value)
    # Pair survival:
    web_result_id = next(
        b["tool_use_id"]
        for b in cast(ToolMessage, sent[1]).content
        if isinstance(b, dict) and b.get("type") == "web_search_tool_result"
    )
    mcp_result_id = next(
        b["tool_use_id"]
        for b in cast(ToolMessage, sent[2]).content
        if isinstance(b, dict) and b.get("type") == "mcp_tool_result"
    )
    assert web_result_id == server_block_id
    assert mcp_result_id == mcp_block_id


def test_client_alignment_skips_server_and_mcp_tool_use_blocks() -> None:
    """Server and MCP tool-use blocks do not position-pair with `tool_calls`."""
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {
                    "type": "server_tool_use",
                    "id": "srv.1",
                    "name": "web_search",
                    "input": {},
                },
                {
                    "type": "mcp_tool_use",
                    "id": "mcp.2",
                    "name": "fetch",
                    "input": {},
                },
                {
                    "type": "tool_use",
                    "id": "client.3",
                    "name": "client_tool",
                    "input": {},
                },
            ],
            tool_calls=[
                {
                    "name": "client_tool",
                    "args": {},
                    "id": "client.3",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv.1",
                    "content": "hits",
                },
            ],
            tool_call_id="srv.1",
        ),
        ToolMessage(
            content=[
                {
                    "type": "mcp_tool_result",
                    "tool_use_id": "mcp.2",
                    "content": "data",
                },
            ],
            tool_call_id="mcp.2",
        ),
        ToolMessage(
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "client.3",
                    "content": "done",
                },
            ],
            tool_call_id="client.3",
        ),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = cast(AIMessage, sent[0])
    server_block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "server_tool_use"
    )
    mcp_block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "mcp_tool_use"
    )
    client_block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    client_call_id = _valid_tool_call_id(ai.tool_calls[0]["id"])
    web_result_id = next(
        b["tool_use_id"]
        for b in cast(ToolMessage, sent[1]).content
        if isinstance(b, dict) and b.get("type") == "web_search_tool_result"
    )
    mcp_result_id = next(
        b["tool_use_id"]
        for b in cast(ToolMessage, sent[2]).content
        if isinstance(b, dict) and b.get("type") == "mcp_tool_result"
    )
    client_result_id = next(
        b["tool_use_id"]
        for b in cast(ToolMessage, sent[3]).content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )

    assert server_block_id == web_result_id == cast(ToolMessage, sent[1]).tool_call_id
    assert mcp_block_id == mcp_result_id == cast(ToolMessage, sent[2]).tool_call_id
    assert client_block_id == client_call_id == client_result_id
    assert cast(ToolMessage, sent[3]).tool_call_id == client_call_id
    assert server_block_id != client_call_id
    assert mcp_block_id != client_call_id


def test_partial_update_only_rewrites_what_changed() -> None:
    """An AIMessage with a clean tool_calls but dirty content block triggers
    rewrite of the content only, not the tool_calls list (partial-update path).
    """
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {"type": "text", "text": "thinking"},
                {
                    "type": "tool_use",
                    "id": "functions.x:0",  # invalid
                    "name": "x",
                    "input": {},
                },
            ],
            tool_calls=[
                # Valid id; should NOT be rewritten by mapping. Drift correction
                # will re-align the content block to use this id.
                {"name": "x", "args": {}, "id": "toolu_canon", "type": "tool_call"},
            ],
        ),
        ToolMessage(content="r", tool_call_id="toolu_canon"),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    ai = cast(AIMessage, sent[0])
    block_id = next(
        b["id"]
        for b in ai.content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    # Drift correction collapses the divergent block id onto the canonical one.
    assert ai.tool_calls[0]["id"] == "toolu_canon"
    assert block_id == "toolu_canon"
    assert cast(ToolMessage, sent[1]).tool_call_id == "toolu_canon"


def test_unrelated_message_subclasses_pass_through() -> None:
    """SystemMessage and HumanMessage are returned untouched even mid-rewrite."""
    sys_msg = SystemMessage(content="you are an agent")
    human = HumanMessage(content="hi")
    messages: list[BaseMessage] = [
        sys_msg,
        human,
        AIMessage(
            content="",
            tool_calls=[
                {"name": "x", "args": {}, "id": "functions.x:0", "type": "tool_call"}
            ],
        ),
        ToolMessage(content="r", tool_call_id="functions.x:0"),
    ]
    middleware, request, captured = _capture(messages)

    middleware.wrap_model_call(request, _handler_factory(captured))

    sent = captured[0].messages
    # Non-AI/non-Tool messages keep their identity.
    assert sent[0] is sys_msg
    assert sent[1] is human


async def test_async_path_rewrites_ids() -> None:
    """`awrap_model_call` rewrites IDs the same way as the sync path."""
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "grep",
                    "args": {},
                    "id": "functions.grep:0",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="r", tool_call_id="functions.grep:0"),
    ]
    middleware, request, captured = _capture(messages)

    async def handler(req: ModelRequest) -> ModelResponse:
        captured.append(req)
        return ModelResponse(result=[AIMessage(content="ok")])

    await middleware.awrap_model_call(request, handler)

    sent = captured[0].messages
    rewritten_id = _valid_tool_call_id(cast(AIMessage, sent[0]).tool_calls[0]["id"])
    assert cast(ToolMessage, sent[1]).tool_call_id == rewritten_id


def test_unsupported_model_ignore_default_skips_silently() -> None:
    """Default `unsupported_model_behavior='ignore'` does not warn or rewrite."""
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "grep",
                    "args": {},
                    "id": "functions.grep:0",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="r", tool_call_id="functions.grep:0"),
    ]
    middleware = AnthropicToolIdSanitizationMiddleware()
    request = _make_request(messages, anthropic=False)
    captured: list[ModelRequest] = []

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        middleware.wrap_model_call(request, _handler_factory(captured))

    assert captured[0].messages is request.messages


def test_unsupported_model_warn_emits_warning() -> None:
    """`unsupported_model_behavior='warn'` warns and skips rewrite."""
    middleware = AnthropicToolIdSanitizationMiddleware(
        unsupported_model_behavior="warn"
    )
    request = _make_request([HumanMessage("hi")], anthropic=False)
    captured: list[ModelRequest] = []

    with pytest.warns(UserWarning, match="only supports Anthropic"):
        middleware.wrap_model_call(request, _handler_factory(captured))


def test_unsupported_model_raise_errors() -> None:
    """`unsupported_model_behavior='raise'` raises `ValueError`."""
    middleware = AnthropicToolIdSanitizationMiddleware(
        unsupported_model_behavior="raise"
    )
    request = _make_request([HumanMessage("hi")], anthropic=False)
    captured: list[ModelRequest] = []

    with pytest.raises(ValueError, match="only supports Anthropic"):
        middleware.wrap_model_call(request, _handler_factory(captured))


def test_invalid_unsupported_model_behavior_rejected() -> None:
    """A typo in `unsupported_model_behavior` raises `ValueError` at construction.

    `Literal` is enforced statically only; without runtime validation a typo
    silently falls through to ignore semantics — a footgun precisely for users
    who opted into `'raise'` to surface bugs.
    """
    with pytest.raises(ValueError, match="unsupported_model_behavior"):
        AnthropicToolIdSanitizationMiddleware(
            unsupported_model_behavior="raies",  # type: ignore[arg-type]
        )


def test_unsupported_model_ignore_logs_debug_breadcrumb(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default `'ignore'` mode emits a debug log so the bypass is observable."""
    middleware = AnthropicToolIdSanitizationMiddleware()
    request = _make_request([HumanMessage("hi")], anthropic=False)
    captured: list[ModelRequest] = []

    with caplog.at_level(
        logging.DEBUG, logger="langchain_anthropic.middleware.tool_id_sanitization"
    ):
        middleware.wrap_model_call(request, _handler_factory(captured))

    assert any(
        "skipped" in record.message and "_FakeModel" in record.message
        for record in caplog.records
    )
