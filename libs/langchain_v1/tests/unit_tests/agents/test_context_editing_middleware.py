"""Tests for the ContextEditingMiddleware."""

from __future__ import annotations

from typing import Iterable, cast

from langchain.agents.middleware.context_editing import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
)
from langchain.agents.middleware.types import AgentState, ModelRequest
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import (
    AIMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime


class _TokenCountingChatModel(FakeChatModel):
    """Fake chat model that counts tokens deterministically for tests."""

    def get_num_tokens_from_messages(
        self,
        messages: list[MessageLikeRepresentation],
        tools: Iterable | None = None,  # noqa: ARG002
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
        return sum(_count_content(block) for block in content)  # type: ignore[arg-type]
    if isinstance(content, dict):
        return len(str(content))
    return len(str(content))


def _make_state_and_request(
    messages: list[AIMessage | ToolMessage],
    *,
    system_prompt: str | None = None,
) -> tuple[AgentState, ModelRequest]:
    model = _TokenCountingChatModel()
    conversation = list(messages)
    state = cast(AgentState, {"messages": conversation})
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

    state, request = _make_state_and_request([ai_message, tool_message])
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call which modifies the request
    middleware.wrap_model_call(request, mock_handler)

    # The request should have been modified in place
    assert request.messages[0].content == ""
    assert request.messages[1].content == "12345"
    assert state["messages"] == request.messages


def test_clear_tool_outputs_and_inputs() -> None:
    tool_call_id = "call-2"
    ai_message = AIMessage(
        content=[
            {"type": "tool_call", "id": tool_call_id, "name": "search", "args": {"query": "foo"}}
        ],
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {"query": "foo"}}],
    )
    tool_message = ToolMessage(content="x" * 200, tool_call_id=tool_call_id)

    state, request = _make_state_and_request([ai_message, tool_message])

    edit = ClearToolUsesEdit(
        trigger=50,
        clear_at_least=10,
        clear_tool_inputs=True,
        keep=0,
        placeholder="[cleared output]",
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call which modifies the request
    middleware.wrap_model_call(request, mock_handler)

    cleared_ai = request.messages[0]
    cleared_tool = request.messages[1]

    assert isinstance(cleared_tool, ToolMessage)
    assert cleared_tool.content == "[cleared output]"
    assert cleared_tool.response_metadata["context_editing"]["cleared"] is True

    assert isinstance(cleared_ai, AIMessage)
    assert cleared_ai.tool_calls[0]["args"] == {}
    context_meta = cleared_ai.response_metadata.get("context_editing")
    assert context_meta is not None
    assert context_meta["cleared_tool_inputs"] == [tool_call_id]

    assert state["messages"] == request.messages


def test_respects_keep_last_tool_results() -> None:
    conversation: list[AIMessage | ToolMessage] = []
    edits = [
        ("call-a", "tool-output-a" * 5),
        ("call-b", "tool-output-b" * 5),
        ("call-c", "tool-output-c" * 5),
    ]

    for call_id, text in edits:
        conversation.append(
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
            )
        )
        conversation.append(ToolMessage(content=text, tool_call_id=call_id))

    state, request = _make_state_and_request(conversation)

    middleware = ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=50,
                keep=1,
                placeholder="[cleared]",
            )
        ],
        token_count_method="model",
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call which modifies the request
    middleware.wrap_model_call(request, mock_handler)

    cleared_messages = [
        msg
        for msg in request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    assert len(cleared_messages) == 2
    assert isinstance(request.messages[-1], ToolMessage)
    assert request.messages[-1].content != "[cleared]"


def test_exclude_tools_prevents_clearing() -> None:
    search_call = "call-search"
    calc_call = "call-calc"

    state, request = _make_state_and_request(
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

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call which modifies the request
    middleware.wrap_model_call(request, mock_handler)

    search_tool = request.messages[1]
    calc_tool = request.messages[3]

    assert isinstance(search_tool, ToolMessage)
    assert search_tool.content == "search-results" * 20

    assert isinstance(calc_tool, ToolMessage)
    assert calc_tool.content == "[cleared]"


def _fake_runtime() -> Runtime:
    return cast(Runtime, object())


async def test_no_edit_when_below_trigger_async() -> None:
    """Test async version of context editing with no edit when below trigger."""
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="12345", tool_call_id=tool_call_id)

    state, request = _make_state_and_request([ai_message, tool_message])
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
    )

    async def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call awrap_model_call which modifies the request
    await middleware.awrap_model_call(request, mock_handler)

    # The request should have been modified in place
    assert request.messages[0].content == ""
    assert request.messages[1].content == "12345"
    assert state["messages"] == request.messages


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

    state, request = _make_state_and_request([ai_message, tool_message])

    edit = ClearToolUsesEdit(
        trigger=50,
        clear_at_least=10,
        clear_tool_inputs=True,
        keep=0,
        placeholder="[cleared output]",
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    async def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call awrap_model_call which modifies the request
    await middleware.awrap_model_call(request, mock_handler)

    cleared_ai = request.messages[0]
    cleared_tool = request.messages[1]

    assert isinstance(cleared_tool, ToolMessage)
    assert cleared_tool.content == "[cleared output]"
    assert cleared_tool.response_metadata["context_editing"]["cleared"] is True

    assert isinstance(cleared_ai, AIMessage)
    assert cleared_ai.tool_calls[0]["args"] == {}
    context_meta = cleared_ai.response_metadata.get("context_editing")
    assert context_meta is not None
    assert context_meta["cleared_tool_inputs"] == [tool_call_id]

    assert state["messages"] == request.messages


async def test_respects_keep_last_tool_results_async() -> None:
    """Test async version respects keep parameter for last tool results."""
    conversation: list[AIMessage | ToolMessage] = []
    edits = [
        ("call-a", "tool-output-a" * 5),
        ("call-b", "tool-output-b" * 5),
        ("call-c", "tool-output-c" * 5),
    ]

    for call_id, text in edits:
        conversation.append(
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
            )
        )
        conversation.append(ToolMessage(content=text, tool_call_id=call_id))

    state, request = _make_state_and_request(conversation)

    middleware = ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=50,
                keep=1,
                placeholder="[cleared]",
            )
        ],
        token_count_method="model",
    )

    async def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call awrap_model_call which modifies the request
    await middleware.awrap_model_call(request, mock_handler)

    cleared_messages = [
        msg
        for msg in request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    assert len(cleared_messages) == 2
    assert isinstance(request.messages[-1], ToolMessage)
    assert request.messages[-1].content != "[cleared]"


async def test_exclude_tools_prevents_clearing_async() -> None:
    """Test async version of excluding tools from clearing."""
    search_call = "call-search"
    calc_call = "call-calc"

    state, request = _make_state_and_request(
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

    async def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call awrap_model_call which modifies the request
    await middleware.awrap_model_call(request, mock_handler)

    search_tool = request.messages[1]
    calc_tool = request.messages[3]

    assert isinstance(search_tool, ToolMessage)
    assert search_tool.content == "search-results" * 20

    assert isinstance(calc_tool, ToolMessage)
    assert calc_tool.content == "[cleared]"


# ==============================================================================
# SystemMessage Tests
# ==============================================================================


def test_handles_system_message_prompt() -> None:
    """Test that middleware handles SystemMessage as system_prompt correctly."""
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="12345", tool_call_id=tool_call_id)

    system_prompt = SystemMessage(content="You are a helpful assistant.")
    state, request = _make_state_and_request(
        [ai_message, tool_message], system_prompt=None
    )
    # Manually set SystemMessage as system_prompt
    request.system_prompt = system_prompt

    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
        token_count_method="model",
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call - should not fail with SystemMessage
    middleware.wrap_model_call(request, mock_handler)

    # Request should have processed without errors
    assert request.system_prompt == system_prompt
    assert isinstance(request.system_prompt, SystemMessage)


def test_does_not_double_wrap_system_message() -> None:
    """Test that middleware doesn't wrap SystemMessage in another SystemMessage."""
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="x" * 100, tool_call_id=tool_call_id)

    system_prompt = SystemMessage(content="Original system prompt")
    state, request = _make_state_and_request(
        [ai_message, tool_message], system_prompt=None
    )
    request.system_prompt = system_prompt

    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
        token_count_method="model",
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    middleware.wrap_model_call(request, mock_handler)

    # System prompt should still be the same SystemMessage, not wrapped
    assert request.system_prompt == system_prompt
    assert isinstance(request.system_prompt, SystemMessage)
    assert request.system_prompt.content == "Original system prompt"


async def test_handles_system_message_prompt_async() -> None:
    """Test async version - middleware handles SystemMessage as system_prompt correctly."""
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="12345", tool_call_id=tool_call_id)

    system_prompt = SystemMessage(content="You are a helpful assistant.")
    state, request = _make_state_and_request(
        [ai_message, tool_message], system_prompt=None
    )
    # Manually set SystemMessage as system_prompt
    request.system_prompt = system_prompt

    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
        token_count_method="model",
    )

    async def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call awrap_model_call - should not fail with SystemMessage
    await middleware.awrap_model_call(request, mock_handler)

    # Request should have processed without errors
    assert request.system_prompt == system_prompt
    assert isinstance(request.system_prompt, SystemMessage)


async def test_does_not_double_wrap_system_message_async() -> None:
    """Test async version - middleware doesn't wrap SystemMessage in another SystemMessage."""
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="x" * 100, tool_call_id=tool_call_id)

    system_prompt = SystemMessage(content="Original system prompt")
    state, request = _make_state_and_request(
        [ai_message, tool_message], system_prompt=None
    )
    request.system_prompt = system_prompt

    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50)],
        token_count_method="model",
    )

    async def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    await middleware.awrap_model_call(request, mock_handler)

    # System prompt should still be the same SystemMessage, not wrapped
    assert request.system_prompt == system_prompt
    assert isinstance(request.system_prompt, SystemMessage)
    assert request.system_prompt.content == "Original system prompt"
