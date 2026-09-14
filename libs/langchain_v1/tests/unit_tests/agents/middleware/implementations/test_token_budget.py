from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from pydantic import Field
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware.token_budget import (
    CHECK_IN_SOURCE,
    DEFAULT_CHECK_IN_PROMPT,
    TokenBudgetExceededError,
    TokenBudgetMiddleware,
    TokenBudgetState,
)
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


@tool
def simple_tool(value: str) -> str:
    """A simple tool."""
    return value


@tool
def ask_user(question: str) -> str:
    """Ask the user a question."""
    return f"answer to {question}"


class FakeUsageModel(FakeToolCallingModel):
    """Fake model that reports `usage_metadata` and records the tools it was given."""

    tokens_per_call: int = 60
    seen_tools: list[list[str] | None] = Field(default_factory=list)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        self.seen_tools.append(None if tools is None else [t["function"]["name"] for t in tools])
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
        message.usage_metadata = {
            "input_tokens": self.tokens_per_call - 10,
            "output_tokens": 10,
            "total_tokens": self.tokens_per_call,
        }
        return result


def _tool_call(call_id: str, name: str = "simple_tool") -> dict[str, Any]:
    args = {"value": "x"} if name == "simple_tool" else {"question": "?"}
    return {"name": name, "args": args, "id": call_id}


def _check_in_messages(messages: list[BaseMessage]) -> list[HumanMessage]:
    return [
        m
        for m in messages
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") == CHECK_IN_SOURCE
    ]


def _make_request(state: TokenBudgetState) -> ModelRequest:
    return ModelRequest(
        model=FakeUsageModel(),
        messages=[HumanMessage("hi")],
        tools=[simple_tool, ask_user],
        tool_choice="auto",
        state=state,
        runtime=Runtime(),
    )


def test_init_validation() -> None:
    with pytest.raises(ValueError, match="At least one limit"):
        TokenBudgetMiddleware()

    with pytest.raises(ValueError, match="turn_token_limit must be a positive integer"):
        TokenBudgetMiddleware(turn_token_limit=0)

    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        TokenBudgetMiddleware(turn_token_limit=10, exit_behavior="stop")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="not both"):
        TokenBudgetMiddleware(
            turn_token_limit=10, exempt_tools=["ask_user"], restricted_tools=["simple_tool"]
        )


def test_before_model_check_in_behavior() -> None:
    middleware = TokenBudgetMiddleware(turn_token_limit=100, thread_token_limit=1000)
    runtime = Runtime()

    # Under budget: nothing happens
    state = TokenBudgetState(messages=[], turn_token_usage=50, thread_token_usage=50)
    assert middleware.before_model(state, runtime) is None

    # Turn budget exceeded: inject the check-in message once
    state = TokenBudgetState(messages=[], turn_token_usage=120, thread_token_usage=120)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert "jump_to" not in result
    assert result["token_budget_check_in_sent"] is True
    (message,) = result["messages"]
    assert isinstance(message, HumanMessage)
    assert message.content == DEFAULT_CHECK_IN_PROMPT
    assert message.additional_kwargs == {"lc_source": CHECK_IN_SOURCE}

    # Check-in already sent but the model kept calling restricted tools: end the run
    state = TokenBudgetState(
        messages=[AIMessage(content="", tool_calls=[_tool_call("1")])],
        turn_token_usage=200,
        thread_token_usage=200,
        token_budget_check_in_sent=True,
    )
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    (message,) = result["messages"]
    assert isinstance(message, AIMessage)
    assert "turn limit (200/100)" in message.content

    # Thread budget exceeded on its own
    state = TokenBudgetState(messages=[], turn_token_usage=0, thread_token_usage=1000)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["token_budget_check_in_sent"] is True


def test_before_model_continues_after_exempt_tool_calls() -> None:
    middleware = TokenBudgetMiddleware(turn_token_limit=100, exempt_tools=["ask_user"])
    runtime = Runtime()

    # Only exempt tools were called: keep going
    state = TokenBudgetState(
        messages=[AIMessage(content="", tool_calls=[_tool_call("1", "ask_user")])],
        turn_token_usage=200,
        token_budget_check_in_sent=True,
        token_budget_exempt_tools=["ask_user"],
    )
    assert middleware.before_model(state, runtime) is None

    # A restricted tool was called alongside an exempt one: end the run
    state = TokenBudgetState(
        messages=[AIMessage(content="", tool_calls=[_tool_call("1", "ask_user"), _tool_call("2")])],
        turn_token_usage=200,
        token_budget_check_in_sent=True,
        token_budget_exempt_tools=["ask_user"],
    )
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"


def test_before_model_end_and_error_behavior() -> None:
    runtime = Runtime()
    state = TokenBudgetState(messages=[], turn_token_usage=150, thread_token_usage=150)

    end_middleware = TokenBudgetMiddleware(turn_token_limit=100, exit_behavior="end")
    result = end_middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "turn limit (150/100)" in result["messages"][0].content

    error_middleware = TokenBudgetMiddleware(turn_token_limit=100, exit_behavior="error")
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        error_middleware.before_model(state, runtime)
    assert "turn limit (150/100)" in str(exc_info.value)
    assert exc_info.value.turn_usage == 150
    assert exc_info.value.turn_limit == 100


def test_custom_check_in_prompt() -> None:
    middleware = TokenBudgetMiddleware(turn_token_limit=10, check_in_prompt="Wrap up now.")
    state = TokenBudgetState(messages=[], turn_token_usage=10)
    result = middleware.before_model(state, Runtime())
    assert result is not None
    assert result["messages"][0].content == "Wrap up now."


def test_wrap_model_call_records_usage_metadata() -> None:
    middleware = TokenBudgetMiddleware(turn_token_limit=1000)
    request = _make_request(
        TokenBudgetState(messages=[], turn_token_usage=5, thread_token_usage=40)
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert req.tools == [simple_tool, ask_user]
        return ModelResponse(
            result=[
                AIMessage(
                    content="ok",
                    usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                )
            ]
        )

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ExtendedModelResponse)
    assert result.model_response.result[0].content == "ok"
    assert result.command is not None
    assert result.command.update == {"turn_token_usage": 30, "thread_token_usage": 65}


def test_wrap_model_call_falls_back_to_token_counter() -> None:
    seen: list[list[BaseMessage]] = []

    def counter(messages: Any) -> int:
        seen.append(list(messages))
        return 42

    middleware = TokenBudgetMiddleware(turn_token_limit=1000, token_counter=counter)
    request = _make_request(TokenBudgetState(messages=[]))

    def handler(_req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="no usage here")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update == {"turn_token_usage": 42, "thread_token_usage": 42}
    # The counter sees the request messages and the response
    assert [m.content for m in seen[0]] == ["hi", "no usage here"]


def test_wrap_model_call_strips_tools_after_check_in() -> None:
    middleware = TokenBudgetMiddleware(turn_token_limit=10)
    request = _make_request(
        TokenBudgetState(messages=[], turn_token_usage=10, token_budget_check_in_sent=True)
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert req.tools == []
        assert req.tool_choice is None
        return ModelResponse(result=[AIMessage(content="summary")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert result.command.update["token_budget_exempt_tools"] == []
    # Original request is left untouched
    assert request.tools == [simple_tool, ask_user]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exempt_tools": ["ask_user"]},
        {"exempt_tools": lambda t: isinstance(t, BaseTool) and t.name == "ask_user"},
        {"restricted_tools": ["simple_tool"]},
        {"restricted_tools": lambda t: isinstance(t, BaseTool) and t.name == "simple_tool"},
    ],
)
def test_wrap_model_call_keeps_exempt_tools_after_check_in(kwargs: dict[str, Any]) -> None:
    middleware = TokenBudgetMiddleware(turn_token_limit=10, **kwargs)
    request = _make_request(
        TokenBudgetState(messages=[], turn_token_usage=10, token_budget_check_in_sent=True)
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert req.tools == [ask_user]
        return ModelResponse(result=[AIMessage(content="summary")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert result.command.update["token_budget_exempt_tools"] == ["ask_user"]


def test_check_in_with_create_agent() -> None:
    """The agent stops calling tools, summarizes and hands control back to the user."""
    model = FakeUsageModel(
        tool_calls=[[_tool_call("1")], [_tool_call("2")], []],
        tokens_per_call=60,
    )
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100)],
    )

    result = agent.invoke({"messages": [HumanMessage("Do the task")]})
    messages = result["messages"]

    # Two tool-calling rounds (60 + 60 tokens) exceed the budget of 100, so the
    # third model call is the check-in.
    check_ins = _check_in_messages(messages)
    assert len(check_ins) == 1
    assert check_ins[0].content == DEFAULT_CHECK_IN_PROMPT

    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    assert len(ai_messages) == 3
    assert ai_messages[-1].tool_calls == []
    assert messages[-1] is ai_messages[-1]

    # The check-in sits right before the final model reply, after the tool results
    assert isinstance(messages[-3], ToolMessage)
    assert messages[-2] is check_ins[0]

    # Tools were bound for the first two calls and removed for the check-in call
    assert model.seen_tools == [["simple_tool"], ["simple_tool"], None]

    # Private state is not exposed in the output
    assert "turn_token_usage" not in result
    assert "thread_token_usage" not in result
    assert "token_budget_check_in_sent" not in result
    assert "token_budget_exempt_tools" not in result


def test_check_in_ends_run_if_model_keeps_calling_tools() -> None:
    """A model that ignores the check-in gets one more chance, then the run is ended."""
    # This fake always emits a tool call, even when no tools are bound.
    model = FakeUsageModel(tool_calls=[[_tool_call("1")]], tokens_per_call=60)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100)],
    )

    result = agent.invoke({"messages": [HumanMessage("Do the task")]})
    messages = result["messages"]

    assert len(_check_in_messages(messages)) == 1
    # Every tool call is answered, so the history stays consistent
    tool_call_ids = [tc["id"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    tool_message_ids = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert tool_call_ids == tool_message_ids
    assert isinstance(messages[-1], AIMessage)
    assert "Token budget exceeded" in messages[-1].content


def test_exempt_tools_with_create_agent() -> None:
    """Exempt tools stay available after the check-in and do not end the run."""
    model = FakeUsageModel(
        tool_calls=[[_tool_call("1")], [_tool_call("2")], [_tool_call("3", "ask_user")], []],
        tokens_per_call=60,
    )
    agent = create_agent(
        model=model,
        tools=[simple_tool, ask_user],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100, exempt_tools=["ask_user"])],
    )

    result = agent.invoke({"messages": [HumanMessage("Do the task")]})
    messages = result["messages"]

    assert len(_check_in_messages(messages)) == 1
    # After the check-in only `ask_user` is bound, and calling it keeps the run going
    assert model.seen_tools == [
        ["simple_tool", "ask_user"],
        ["simple_tool", "ask_user"],
        ["ask_user"],
        ["ask_user"],
    ]
    assert isinstance(messages[-2], ToolMessage)
    assert messages[-2].tool_call_id == "3"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].tool_calls == []
    assert "Token budget exceeded" not in messages[-1].content


def test_restricted_tools_with_create_agent() -> None:
    """Calling a restricted tool after the check-in ends the run."""
    model = FakeUsageModel(
        tool_calls=[[_tool_call("1")], [_tool_call("2")], [_tool_call("3")], []],
        tokens_per_call=60,
    )
    agent = create_agent(
        model=model,
        tools=[simple_tool, ask_user],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100, restricted_tools=["simple_tool"])],
    )

    result = agent.invoke({"messages": [HumanMessage("Do the task")]})
    messages = result["messages"]

    assert len(_check_in_messages(messages)) == 1
    assert model.seen_tools == [
        ["simple_tool", "ask_user"],
        ["simple_tool", "ask_user"],
        ["ask_user"],
    ]
    assert isinstance(messages[-1], AIMessage)
    assert "Token budget exceeded: turn limit (180/100)" in messages[-1].content


def test_end_behavior_with_create_agent() -> None:
    model = FakeUsageModel(tool_calls=[[_tool_call("1")]], tokens_per_call=60)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100, exit_behavior="end")],
    )

    result = agent.invoke({"messages": [HumanMessage("Do the task")]})
    messages = result["messages"]

    assert _check_in_messages(messages) == []
    assert isinstance(messages[-1], AIMessage)
    assert "turn limit (120/100)" in messages[-1].content
    # Two model calls, then the run was ended
    assert len(model.seen_tools) == 2


def test_error_behavior_with_create_agent() -> None:
    model = FakeUsageModel(tool_calls=[[_tool_call("1")]], tokens_per_call=60)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100, exit_behavior="error")],
    )

    with pytest.raises(TokenBudgetExceededError, match=r"turn limit \(120/100\)"):
        agent.invoke({"messages": [HumanMessage("Do the task")]})


def test_turn_budget_resets_between_runs() -> None:
    model = FakeUsageModel(
        tool_calls=[
            [_tool_call("1")],
            [_tool_call("2")],
            [],
            [_tool_call("3")],
            [_tool_call("4")],
            [],
        ],
        tokens_per_call=60,
    )
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100)],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}

    first = agent.invoke({"messages": [HumanMessage("First task")]}, config)
    assert len(_check_in_messages(first["messages"])) == 1

    # A fresh run gets a fresh turn budget: it can make two tool calls again
    second = agent.invoke({"messages": [HumanMessage("Second task")]}, config)
    assert len(_check_in_messages(second["messages"])) == 2
    new_ai_messages = [
        m for m in second["messages"][len(first["messages"]) :] if isinstance(m, AIMessage)
    ]
    assert len(new_ai_messages) == 3
    assert new_ai_messages[-1].tool_calls == []
    # Six model calls in total, tools removed only for the two check-in calls
    assert [t is None for t in model.seen_tools] == [False, False, True, False, False, True]


def test_thread_budget_persists_across_runs() -> None:
    model = FakeUsageModel(
        tool_calls=[[_tool_call("1")], [], [_tool_call("2")], []],
        tokens_per_call=60,
    )
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(thread_token_limit=150)],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}

    # Run 1 uses 120 tokens: under the thread budget, no check-in
    first = agent.invoke({"messages": [HumanMessage("First task")]}, config)
    assert _check_in_messages(first["messages"]) == []

    # Run 2 starts at 120, makes one call (180 >= 150) and then checks in
    second = agent.invoke({"messages": [HumanMessage("Second task")]}, config)
    check_ins = _check_in_messages(second["messages"])
    assert len(check_ins) == 1
    assert isinstance(second["messages"][-1], AIMessage)
    assert second["messages"][-2] is check_ins[0]


async def test_check_in_with_create_agent_async() -> None:
    model = FakeUsageModel(
        tool_calls=[[_tool_call("1")], [_tool_call("2")], []],
        tokens_per_call=60,
    )
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Do the task")]})
    messages = result["messages"]

    check_ins = _check_in_messages(messages)
    assert len(check_ins) == 1
    assert messages[-2] is check_ins[0]
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].tool_calls == []
    assert model.seen_tools[-1] is None


async def test_exempt_tools_with_create_agent_async() -> None:
    model = FakeUsageModel(
        tool_calls=[[_tool_call("1")], [_tool_call("2")], [_tool_call("3", "ask_user")], []],
        tokens_per_call=60,
    )
    agent = create_agent(
        model=model,
        tools=[simple_tool, ask_user],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100, exempt_tools=["ask_user"])],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Do the task")]})
    messages = result["messages"]

    assert len(_check_in_messages(messages)) == 1
    assert model.seen_tools[-2:] == [["ask_user"], ["ask_user"]]
    assert isinstance(messages[-1], AIMessage)
    assert "Token budget exceeded" not in messages[-1].content


async def test_error_behavior_with_create_agent_async() -> None:
    model = FakeUsageModel(tool_calls=[[_tool_call("1")]], tokens_per_call=60)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(turn_token_limit=100, exit_behavior="error")],
    )

    with pytest.raises(TokenBudgetExceededError):
        await agent.ainvoke({"messages": [HumanMessage("Do the task")]})
