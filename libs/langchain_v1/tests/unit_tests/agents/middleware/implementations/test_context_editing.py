"""Tests for the ContextEditingMiddleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import override

from langchain.agents.factory import create_agent
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
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime


class _TokenCountingChatModel(FakeChatModel):
    """Fake chat model that counts tokens deterministically for tests."""

    @override
    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> int:
        return sum(_count_message_tokens(message) for message in messages)


def _count_message_tokens(message: MessageLikeRepresentation) -> int:
    if isinstance(message, (AIMessage, ToolMessage)):
        return _count_content(cast("MessageLikeRepresentation", message.content))
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


def test_wrap_model_call_persists_edit_matching_model_request() -> None:
    """The persisted `Command` must reuse the exact edit decided for the real request.

    Regression test for a review comment on #37815: a persistence pass that
    recomputed the edit with a *different* token counter than the one used
    to build the outgoing model request could disagree about which messages
    to clear (e.g. under `token_count_method="model"`, where the model's
    exact count over messages + system prompt + tools can differ from the
    approximate, messages-only count). Persisting the identical objects that
    were actually sent to the model rules that out structurally.
    """
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="x" * 200, tool_call_id=tool_call_id)

    _state, request = _make_state_and_request([ai_message, tool_message])
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50, keep=0, placeholder="[cleared]")],
        token_count_method="model",  # noqa: S106
    )

    captured_request: ModelRequest | None = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(request, mock_handler)

    assert captured_request is not None
    sent_tool_message = captured_request.messages[1]
    assert isinstance(sent_tool_message, ToolMessage)
    assert sent_tool_message.content == "[cleared]"

    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    persisted = result.command.update["messages"]
    assert len(persisted) == 1
    # The persisted message is the identical object sent to the model, not a
    # second, independently-decided copy.
    assert persisted[0] is sent_tool_message

    # The original state's messages are untouched.
    assert tool_message.content == "x" * 200


def test_wrap_model_call_returns_plain_response_when_no_edit_applied() -> None:
    """Below trigger, `wrap_model_call` must not wrap the response in a `Command`."""
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="12345", tool_call_id=tool_call_id)

    _state, request = _make_state_and_request([ai_message, tool_message])
    middleware = ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=50)])

    def mock_handler(_req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(result, ModelResponse)


def test_wrap_model_call_skips_persistence_for_length_changing_edit() -> None:
    """A length-changing custom `ContextEdit` must not crash persistence.

    Regression test for a review comment on #37815: `ContextEdit.apply` is
    only required to mutate the message list in place — it isn't required to
    preserve length (e.g. a custom strategy could `messages.pop(0)`). Pairing
    up messages by index to detect changes would previously raise
    `ValueError` from a length-mismatched `zip(..., strict=True)`. The edited,
    shortened request must still reach the model, just without persistence
    for that edit.
    """

    class DropFirstMessageEdit:
        """Custom edit that removes the oldest message instead of replacing it."""

        def apply(self, messages: list[AnyMessage], *, count_tokens: Any) -> None:
            del count_tokens
            if messages:
                messages.pop(0)

    ai_message = AIMessage(content="", tool_calls=[])
    tool_message = ToolMessage(content="x" * 200, tool_call_id="call-1")

    _state, request = _make_state_and_request([ai_message, tool_message])
    middleware = ContextEditingMiddleware(edits=[DropFirstMessageEdit()])

    captured_request: ModelRequest | None = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(request, mock_handler)

    assert captured_request is not None
    # The model still receives the shortened request.
    assert captured_request.messages == [tool_message]
    # No persistence is attempted for a length-changing edit.
    assert isinstance(result, ModelResponse)


def test_persisted_edit_is_idempotent_across_turns() -> None:
    """Once persisted, a later turn must not re-clear an already-cleared message.

    Simulates the checkpointer round trip: apply `wrap_model_call`'s
    persisted `Command` update back onto the message list (as `add_messages`
    would), then run `wrap_model_call` again on the grown conversation.
    """
    tool_call_id = "call-1"
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
    )
    tool_message = ToolMessage(content="x" * 200, tool_call_id=tool_call_id)

    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=50, keep=0, placeholder="[cleared]")],
    )

    def mock_handler(_req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    _state, request = _make_state_and_request([ai_message, tool_message])
    first_result = middleware.wrap_model_call(request, mock_handler)
    assert isinstance(first_result, ExtendedModelResponse)
    assert first_result.command is not None
    assert first_result.command.update is not None

    # Apply the persisted update, as the checkpointer/`add_messages` would.
    messages = list(request.messages)
    messages[1] = first_result.command.update["messages"][0]

    # A second call against the already-persisted, cleared conversation
    # changes nothing further.
    _state, second_request = _make_state_and_request(cast("list[Any]", messages))
    second_result = middleware.wrap_model_call(second_request, mock_handler)
    assert isinstance(second_result, ModelResponse)


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


def test_end_to_end_with_checkpointer_clears_persist_across_turns() -> None:
    """End-to-end regression test for #37815 against a real checkpointer.

    Mirrors the issue's repro: each turn appends a large tool result to a
    persistent (`InMemorySaver`-backed) conversation. Once the trigger is
    crossed, older tool results must show up as cleared *in the checkpointed
    state itself* — not just in the transient request sent to the model —
    and stay cleared rather than needing to be reprocessed every turn.
    """

    @tool
    def big_tool(query: str) -> str:  # noqa: ARG001
        """Return a large payload."""
        return "x" * 2_000

    model = FakeToolCallingModel(
        tool_calls=[
            call
            for i in range(4)
            for call in (
                [ToolCall(name="big_tool", args={"query": f"q{i}"}, id=f"call_{i}")],
                [],
            )
        ]
    )
    middleware = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=100, keep=1, placeholder="[cleared]")],
    )
    agent = create_agent(
        model=model,
        tools=[big_tool],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "context-editing-e2e"}}

    for turn in range(4):
        agent.invoke({"messages": [HumanMessage(f"turn {turn}")]}, config=config)

    final_state = agent.get_state(config).values
    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]

    assert len(tool_messages) == 4
    # All but the most recent tool result are cleared *in checkpointed state*.
    assert [m.content for m in tool_messages[:-1]] == ["[cleared]"] * 3
    assert tool_messages[-1].content == "x" * 2_000
    assert all(
        m.response_metadata.get("context_editing", {}).get("cleared") for m in tool_messages[:-1]
    )
