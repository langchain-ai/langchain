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


def test_new_api_with_tokens_trigger_and_messages_keep() -> None:
    """Test new API with token-based trigger and message-based keep."""
    conversation: list[AIMessage | ToolMessage] = []
    for i in range(5):
        call_id = f"call-{i}"
        conversation.append(
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
            )
        )
        conversation.append(ToolMessage(content="x" * 50, tool_call_id=call_id))

    state, request = _make_state_and_request(conversation)

    edit = ClearToolUsesEdit(
        trigger=("tokens", 100),
        keep=("messages", 2),
        placeholder="[cleared]",
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    middleware.wrap_model_call(request, mock_handler)

    # Count how many tool messages were cleared
    cleared_messages = [
        msg
        for msg in request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    # Should clear 3 out of 5 tool messages (keep 2)
    assert len(cleared_messages) == 3

    # Last 2 tool messages should not be cleared
    tool_messages = [msg for msg in request.messages if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 5
    assert tool_messages[-1].content != "[cleared]"
    assert tool_messages[-2].content != "[cleared]"


def test_new_api_with_messages_trigger() -> None:
    """Test new API with message count trigger."""
    conversation: list[AIMessage | ToolMessage] = []
    for i in range(10):
        call_id = f"call-{i}"
        conversation.append(
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
            )
        )
        conversation.append(ToolMessage(content="result", tool_call_id=call_id))

    state, request = _make_state_and_request(conversation)

    # Trigger when we have 15 or more messages
    edit = ClearToolUsesEdit(
        trigger=("messages", 15),
        keep=("messages", 3),
        placeholder="[cleared]",
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    middleware.wrap_model_call(request, mock_handler)

    # Should trigger because we have 20 messages (10 AI + 10 Tool)
    cleared_messages = [
        msg
        for msg in request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    # Should clear 7 out of 10 tool messages (keep 3)
    assert len(cleared_messages) == 7


def test_new_api_with_multiple_triggers() -> None:
    """Test new API with multiple trigger conditions."""
    conversation: list[AIMessage | ToolMessage] = []
    for i in range(3):
        call_id = f"call-{i}"
        conversation.append(
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
            )
        )
        conversation.append(ToolMessage(content="x" * 100, tool_call_id=call_id))

    state, request = _make_state_and_request(conversation)

    # Trigger on either message count OR token count
    edit = ClearToolUsesEdit(
        trigger=[("messages", 5), ("tokens", 200)],
        keep=("messages", 1),
        placeholder="[cleared]",
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    middleware.wrap_model_call(request, mock_handler)

    # Should trigger because we have 6 messages (>5) and also >200 tokens
    cleared_messages = [
        msg
        for msg in request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    # Should clear 2 out of 3 tool messages (keep 1)
    assert len(cleared_messages) == 2


def test_new_api_fractional_trigger_with_model_profile() -> None:
    """Test new API with fractional trigger using model profile."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from langchain_model_profiles import ModelProfile

    class ProfileModel(_TokenCountingChatModel):
        @property
        def profile(self) -> "ModelProfile":
            return {"max_input_tokens": 500}

    conversation: list[AIMessage | ToolMessage] = []
    for i in range(10):
        call_id = f"call-{i}"
        conversation.append(
            AIMessage(
                content="x" * 50,
                tool_calls=[{"id": call_id, "name": "tool", "args": {"input": call_id}}],
            )
        )
        conversation.append(ToolMessage(content="x" * 50, tool_call_id=call_id))

    model = ProfileModel()
    state = cast("AgentState", {"messages": conversation})
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=conversation,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=_fake_runtime(),
        model_settings={},
    )

    # Trigger when we reach 80% of model's max input tokens
    # Total tokens = 1000 (10 AI messages * 50 + 10 tool messages * 50)
    # Max input tokens = 500
    # 80% of 500 = 400 tokens -> should trigger because 1000 > 400
    edit = ClearToolUsesEdit(
        trigger=("fraction", 0.8),
        keep=("messages", 3),
        placeholder="[cleared]",
        model=model,
    )
    middleware = ContextEditingMiddleware(edits=[edit])

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    middleware.wrap_model_call(request, mock_handler)

    cleared_messages = [
        msg
        for msg in request.messages
        if isinstance(msg, ToolMessage) and msg.content == "[cleared]"
    ]

    # Should clear some messages (keep 3)
    assert len(cleared_messages) == 7


def test_validation_errors() -> None:
    """Test that validation errors are raised for invalid parameters."""
    import pytest

    # Invalid fraction (> 1)
    with pytest.raises(ValueError, match="Fractional trigger values must be between 0 and 1"):
        ClearToolUsesEdit(trigger=("fraction", 1.5))

    # Invalid fraction (< 0)
    with pytest.raises(ValueError, match="Fractional trigger values must be between 0 and 1"):
        ClearToolUsesEdit(trigger=("fraction", 0))

    # Invalid trigger (0 tokens)
    with pytest.raises(ValueError, match="trigger thresholds must be greater than 0"):
        ClearToolUsesEdit(trigger=("tokens", 0))

    # Invalid trigger (negative)
    with pytest.raises(ValueError, match="trigger thresholds must be greater than 0"):
        ClearToolUsesEdit(trigger=("messages", -1))

    # Invalid keep (negative)
    with pytest.raises(ValueError, match="keep values must be non-negative"):
        ClearToolUsesEdit(keep=("messages", -1))

    # Unsupported context size type
    with pytest.raises(ValueError, match="Unsupported context size type"):
        ClearToolUsesEdit(trigger=("invalid", 100))  # type: ignore[arg-type]


def test_deprecation_warnings() -> None:
    """Test that deprecation warnings are raised for old API."""
    import pytest

    with pytest.warns(DeprecationWarning, match="Passing trigger as int is deprecated"):
        ClearToolUsesEdit(trigger=100)

    with pytest.warns(DeprecationWarning, match="Passing keep as int is deprecated"):
        ClearToolUsesEdit(keep=5)
