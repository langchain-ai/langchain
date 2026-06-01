"""Tests for the ContextEditingMiddleware."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from typing_extensions import override

from langchain.agents.middleware.context_editing import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
)
from langchain.agents.middleware.types import (
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest
    from langgraph.runtime import Runtime


class _TokenCountingChatModel(FakeChatModel):
    """Fake chat model that counts tokens deterministically for tests."""

    @override
    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence | None = None,
    ) -> int:
        return sum(_count_message_tokens(message) for message in messages)


def _count_message_tokens(message: MessageLikeRepresentation) -> int:
    if isinstance(message, (AIMessage, ToolMessage)):
        return _count_content(message.content)
    if isinstance(message, str):
        return len(message)
    return len(str(message))


def _count_content(content: MessageLikeRepresentation) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_count_content(block) for block in content)
    if isinstance(content, dict):
        return len(str(content))
    return len(str(content))


def _make_state_and_request(
    messages: list[AIMessage | ToolMessage],
    *,
    system_prompt: str | None = None,
) -> tuple[AgentState[Any], ModelRequest]:
    model = _TokenCountingChatModel()
    conversation: list[AnyMessage] = list(messages)
    state = cast("AgentState[Any]", {"messages": conversation})
    request = ModelRequest(
        model=model,
        system_prompt=system_prompt,
        messages=conversation,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=_fake_runtime(),
        model_settings={},
    )
    return state, request


def test_no_edit_when_below_trigger() -> None:
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="12345", tool_call_id=tool_call_id)

    _state, request = _make_state_and_request([ai_message, tool_message])
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
    )

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call wrap_model_call which creates a new request
    middleware.wrap_model_call(request, mock_handler)

    # The modified request passed to handler should be the same since no edits applied
    assert modified_request is not None
    assert modified_request.messages[0].content == ""
    assert modified_request.messages[1].content == "12345"
    # Original request should be unchanged
    assert request.messages[0].content == ""
    assert request.messages[1].content == "12345"


def test_clear_tool_outputs_and_inputs() -> None:
    tool_call_id = "call-2"
    ai_message = AIMessage(
        content=[
            {"type": "tool_call", "id": tool_call_id, "name": "search", "args": {"query": "foo"}}
        ],
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {"query": "foo"}}],
    )
    tool_message = ToolMessage(content="x" * 200, tool_call_id=tool_call_id)

    _state, request = _make_state_and_request([ai_message, tool_message])

    edit = ClearToolUsesEdit(
        trigger=50,
        clear_at_least=10,
        clear_tool_inputs=True,
        keep=0,
        placeholder="[cleared output]",
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call wrap_model_call which creates a new request with edits
    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    cleared_ai = modified_request.messages[0]
    cleared_tool = modified_request.messages[1]

    assert isinstance(cleared_tool, ToolMessage)
    assert cleared_tool.content == "[cleared output]"
    assert cleared_tool.response_metadata["context_editing"]["cleared"] is True

    assert isinstance(cleared_ai, AIMessage)
    assert cleared_ai.tool_calls[0]["args"] == {}
    context_meta = cleared_ai.response_metadata.get("context_editing")
    assert context_meta is not None
    assert context_meta["cleared_tool_inputs"] == [tool_call_id]

    # Original request should be unchanged
    request_ai_message = request.messages[0]
    assert isinstance(request_ai_message, AIMessage)
    assert request_ai_message.tool_calls[0]["args"] == {"query": "foo"}
    assert request.messages[1].content == "x" * 200


def test_respects_keep_last_tool_results() -> None:
    conversation: list[AIMessage | ToolMessage] = []
    edits = [
        ("call-a", "tool-output-a" * 5),
        ("call-b", "tool-output-b" * 5),
        ("call-c", "tool-output-c" * 5),
    ]

    for call_id, text in edits:
        conversation.extend(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
                ),
                ToolMessage(content=text, tool_call_id=call_id),
            )
        )

    _state, request = _make_state_and_request(conversation)

    middleware = ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=50,
                keep=1,
                placeholder="[cleared]",
            )
        ],
        token_count_method="model",  # noqa: S106
    )

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call wrap_model_call which creates a new request with edits
    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    cleared_messages = [
        msg
        for msg in modified_request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    assert len(cleared_messages) == 2
    assert isinstance(modified_request.messages[-1], ToolMessage)
    assert modified_request.messages[-1].content != "[cleared]"


def test_exclude_tools_prevents_clearing() -> None:
    search_call = "call-search"
    calc_call = "call-calc"

    _state, request = _make_state_and_request(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": search_call, "name": "search", "args": {"query": "foo"}}],
            ),
            ToolMessage(content="search-results" * 20, tool_call_id=search_call),
            AIMessage(
                content="",
                tool_calls=[{"id": calc_call, "name": "calculator", "args": {"a": 1, "b": 2}}],
            ),
            ToolMessage(content="42", tool_call_id=calc_call),
        ]
    )

    middleware = ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=50,
                clear_at_least=10,
                keep=0,
                exclude_tools=("search",),
                placeholder="[cleared]",
            )
        ],
    )

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call wrap_model_call which creates a new request with edits
    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    search_tool = modified_request.messages[1]
    calc_tool = modified_request.messages[3]

    assert isinstance(search_tool, ToolMessage)
    assert search_tool.content == "search-results" * 20

    assert isinstance(calc_tool, ToolMessage)
    assert calc_tool.content == "[cleared]"


def _fake_runtime() -> Runtime:
    return cast("Runtime", object())


async def test_no_edit_when_below_trigger_async() -> None:
    """Test async version of context editing with no edit when below trigger."""
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="12345", tool_call_id=tool_call_id)

    _state, request = _make_state_and_request([ai_message, tool_message])
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
    )

    modified_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call awrap_model_call which creates a new request
    await middleware.awrap_model_call(request, mock_handler)

    # The modified request passed to handler should be the same since no edits applied
    assert modified_request is not None
    assert modified_request.messages[0].content == ""
    assert modified_request.messages[1].content == "12345"
    # Original request should be unchanged
    assert request.messages[0].content == ""
    assert request.messages[1].content == "12345"


async def test_clear_tool_outputs_and_inputs_async() -> None:
    """Test async version of clearing tool outputs and inputs."""
    tool_call_id = "call-2"
    ai_message = AIMessage(
        content=[
            {"type": "tool_call", "id": tool_call_id, "name": "search", "args": {"query": "foo"}}
        ],
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {"query": "foo"}}],
    )
    tool_message = ToolMessage(content="x" * 200, tool_call_id=tool_call_id)

    _state, request = _make_state_and_request([ai_message, tool_message])

    edit = ClearToolUsesEdit(
        trigger=50,
        clear_at_least=10,
        clear_tool_inputs=True,
        keep=0,
        placeholder="[cleared output]",
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    modified_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call awrap_model_call which creates a new request with edits
    await middleware.awrap_model_call(request, mock_handler)

    assert modified_request is not None
    cleared_ai = modified_request.messages[0]
    cleared_tool = modified_request.messages[1]

    assert isinstance(cleared_tool, ToolMessage)
    assert cleared_tool.content == "[cleared output]"
    assert cleared_tool.response_metadata["context_editing"]["cleared"] is True

    assert isinstance(cleared_ai, AIMessage)
    assert cleared_ai.tool_calls[0]["args"] == {}
    context_meta = cleared_ai.response_metadata.get("context_editing")
    assert context_meta is not None
    assert context_meta["cleared_tool_inputs"] == [tool_call_id]

    # Original request should be unchanged
    request_ai_message = request.messages[0]
    assert isinstance(request_ai_message, AIMessage)
    assert request_ai_message.tool_calls[0]["args"] == {"query": "foo"}
    assert request.messages[1].content == "x" * 200


async def test_respects_keep_last_tool_results_async() -> None:
    """Test async version respects keep parameter for last tool results."""
    conversation: list[AIMessage | ToolMessage] = []
    edits = [
        ("call-a", "tool-output-a" * 5),
        ("call-b", "tool-output-b" * 5),
        ("call-c", "tool-output-c" * 5),
    ]

    for call_id, text in edits:
        conversation.extend(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
                ),
                ToolMessage(content=text, tool_call_id=call_id),
            )
        )

    _state, request = _make_state_and_request(conversation)

    middleware = ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=50,
                keep=1,
                placeholder="[cleared]",
            )
        ],
        token_count_method="model",  # noqa: S106
    )

    modified_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call awrap_model_call which creates a new request with edits
    await middleware.awrap_model_call(request, mock_handler)

    assert modified_request is not None
    cleared_messages = [
        msg
        for msg in modified_request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    assert len(cleared_messages) == 2
    assert isinstance(modified_request.messages[-1], ToolMessage)
    assert modified_request.messages[-1].content != "[cleared]"


async def test_exclude_tools_prevents_clearing_async() -> None:
    """Test async version of excluding tools from clearing."""
    search_call = "call-search"
    calc_call = "call-calc"

    _state, request = _make_state_and_request(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": search_call, "name": "search", "args": {"query": "foo"}}],
            ),
            ToolMessage(content="search-results" * 20, tool_call_id=search_call),
            AIMessage(
                content="",
                tool_calls=[{"id": calc_call, "name": "calculator", "args": {"a": 1, "b": 2}}],
            ),
            ToolMessage(content="42", tool_call_id=calc_call),
        ]
    )

    middleware = ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=50,
                clear_at_least=10,
                keep=0,
                exclude_tools=("search",),
                placeholder="[cleared]",
            )
        ],
    )

    modified_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call awrap_model_call which creates a new request with edits
    await middleware.awrap_model_call(request, mock_handler)

    assert modified_request is not None
    search_tool = modified_request.messages[1]
    calc_tool = modified_request.messages[3]

    assert isinstance(search_tool, ToolMessage)
    assert search_tool.content == "search-results" * 20

    assert isinstance(calc_tool, ToolMessage)
    assert calc_tool.content == "[cleared]"


def _conversation_with_tool_results(
    call_ids: tuple[str, ...],
    *,
    content_length: int,
) -> list[AIMessage | ToolMessage]:
    conversation: list[AIMessage | ToolMessage] = []
    for call_id in call_ids:
        conversation.extend(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"id": call_id, "name": "tool", "args": {}}],
                ),
                ToolMessage(content="x" * content_length, tool_call_id=call_id),
            )
        )
    return conversation


def test_persists_last_effective_count_after_edit_sync() -> None:
    """When the edit fires, the post-edit effective token count is persisted."""
    conversation = _conversation_with_tool_results(
        ("call-a", "call-b", "call-c"), content_length=100
    )
    _state, request = _make_state_and_request(conversation)
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=150, keep=1, placeholder="[cleared]")],
        token_count_method="model",  # noqa: S106
    )

    captured: dict[str, ModelRequest] = {}

    def mock_handler(req: ModelRequest) -> ModelResponse:
        captured["req"] = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ExtendedModelResponse)
    assert response.command is not None
    model = _TokenCountingChatModel()
    edited_count = model.get_num_tokens_from_messages(list(captured["req"].messages))
    raw_count = model.get_num_tokens_from_messages(list(request.messages))
    # The edit cleared tool outputs, so the persisted count is below the raw count.
    assert edited_count < raw_count
    assert response.command.update["_last_effective_count"] == edited_count


async def test_persists_last_effective_count_after_edit_async() -> None:
    """Async: when the edit fires, the post-edit effective token count is persisted."""
    conversation = _conversation_with_tool_results(
        ("call-a", "call-b", "call-c"), content_length=100
    )
    _state, request = _make_state_and_request(conversation)
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=150, keep=1, placeholder="[cleared]")],
        token_count_method="model",  # noqa: S106
    )

    captured: dict[str, ModelRequest] = {}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        captured["req"] = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ExtendedModelResponse)
    assert response.command is not None
    model = _TokenCountingChatModel()
    edited_count = model.get_num_tokens_from_messages(list(captured["req"].messages))
    raw_count = model.get_num_tokens_from_messages(list(request.messages))
    assert edited_count < raw_count
    assert response.command.update["_last_effective_count"] == edited_count


def test_below_trigger_writes_raw_count() -> None:
    """When no edit fires, the raw token count is still persisted as a baseline."""
    tool_call_id = "call-1"
    conversation: list[AIMessage | ToolMessage] = [
        AIMessage(content="", tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}]),
        ToolMessage(content="12345", tool_call_id=tool_call_id),
    ]
    _state, request = _make_state_and_request(conversation)
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=1000, keep=3, placeholder="[cleared]")],
        token_count_method="model",  # noqa: S106
    )

    captured: dict[str, ModelRequest] = {}

    def mock_handler(req: ModelRequest) -> ModelResponse:
        captured["req"] = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ExtendedModelResponse)
    assert response.command is not None
    # Nothing was cleared, so the handler still sees the original content.
    assert captured["req"].messages[1].content == "12345"
    model = _TokenCountingChatModel()
    raw_count = model.get_num_tokens_from_messages(list(request.messages))
    assert response.command.update["_last_effective_count"] == raw_count


def test_warns_when_effective_count_exceeds_trigger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When clearing cannot bring the conversation under budget, warn the caller."""
    tool_call_id = "call-1"
    conversation: list[AIMessage | ToolMessage] = [
        AIMessage(content="", tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}]),
        ToolMessage(content="x" * 100, tool_call_id=tool_call_id),
    ]
    _state, request = _make_state_and_request(conversation)
    # ``keep`` exceeds the number of clearable tool results, so nothing is cleared
    # and the effective count stays above the trigger.
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50, keep=3, placeholder="[cleared]")],
        token_count_method="model",  # noqa: S106
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    with caplog.at_level(logging.WARNING, logger="langchain.agents.middleware.context_editing"):
        response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ExtendedModelResponse)
    assert response.command is not None
    assert response.command.update["_last_effective_count"] == 100
    assert any("exceed" in record.getMessage().lower() for record in caplog.records)
