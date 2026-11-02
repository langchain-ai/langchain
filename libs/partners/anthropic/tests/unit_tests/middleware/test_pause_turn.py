"""Tests for the Anthropic `pause_turn` continuation middleware."""

from collections.abc import Iterator
from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.runtime import Runtime

from langchain_anthropic.middleware import (
    AnthropicPauseTurnMiddleware,
    PauseTurnLimitExceededError,
)

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search"}


def _paused(index: int) -> AIMessage:
    return AIMessage(
        content=[
            {"type": "text", "text": "Let me search for that."},
            {
                "type": "server_tool_use",
                "id": f"srvtoolu_{index}",
                "name": "web_search",
                "input": {"query": "langgraph latest release"},
            },
        ],
        response_metadata={"stop_reason": "pause_turn"},
        id=f"paused-{index}",
    )


def _final() -> AIMessage:
    return AIMessage(
        content="The latest release is 1.2.",
        response_metadata={"stop_reason": "end_turn"},
        id="final",
    )


def _request(messages: list[AnyMessage]) -> ModelRequest:
    return ModelRequest(
        model=cast(BaseChatModel, object()),
        messages=messages,
        system_prompt="You are a research assistant.",
        tool_choice=None,
        tools=[WEB_SEARCH],
        response_format=None,
        state={"messages": messages},
        runtime=cast(Runtime, object()),
        model_settings={"max_tokens": 1024},
    )


class _ScriptedHandler:
    """Answers each model call with the next scripted message and records the call."""

    def __init__(self, replies: list[AIMessage]) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(result=[self.replies.pop(0)])

    async def acall(self, request: ModelRequest) -> ModelResponse:
        return self(request)


def test_sends_paused_turn_back_unchanged() -> None:
    question = HumanMessage("What is the latest LangGraph release?")
    request = _request([question])
    paused = _paused(1)
    handler = _ScriptedHandler([paused, _final()])

    response = AnthropicPauseTurnMiddleware().wrap_model_call(request, handler)

    assert len(handler.requests) == 2
    continuation = handler.requests[1]
    assert continuation.messages == [question, paused]
    assert continuation.tools == request.tools
    assert continuation.system_message == request.system_message
    assert continuation.model_settings == request.model_settings
    assert isinstance(response, ModelResponse)
    assert response.result == [paused, _final()]


def test_continues_until_the_turn_ends() -> None:
    question = HumanMessage("What is the latest LangGraph release?")
    first, second = _paused(1), _paused(2)
    handler = _ScriptedHandler([first, second, _final()])

    response = AnthropicPauseTurnMiddleware().wrap_model_call(
        _request([question]), handler
    )

    assert [r.messages for r in handler.requests] == [
        [question],
        [question, first],
        [question, first, second],
    ]
    assert isinstance(response, ModelResponse)
    assert response.result == [first, second, _final()]


@pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use", "max_tokens", None])
def test_turn_that_is_not_paused_is_returned_as_is(stop_reason: str | None) -> None:
    reply = AIMessage(
        content="Done.",
        response_metadata={} if stop_reason is None else {"stop_reason": stop_reason},
    )
    handler = _ScriptedHandler([reply, _final()])

    response = AnthropicPauseTurnMiddleware().wrap_model_call(
        _request([HumanMessage("Hi")]), handler
    )

    assert len(handler.requests) == 1
    assert isinstance(response, ModelResponse)
    assert response.result == [reply]


def test_raises_once_the_bound_is_spent() -> None:
    turn = [_paused(1), _paused(2), _paused(3)]
    handler = _ScriptedHandler([*turn, _final()])

    with pytest.raises(PauseTurnLimitExceededError, match="2 continuations") as error:
        AnthropicPauseTurnMiddleware(max_continuations=2).wrap_model_call(
            _request([HumanMessage("Hi")]), handler
        )

    assert error.value.max_continuations == 2
    assert error.value.messages == turn
    assert all(m is r for m, r in zip(error.value.messages, turn, strict=True))
    assert len(handler.requests) == 3


def test_turn_that_ends_on_the_last_allowed_continuation_does_not_raise() -> None:
    handler = _ScriptedHandler([_paused(1), _paused(2), _final()])

    response = AnthropicPauseTurnMiddleware(max_continuations=2).wrap_model_call(
        _request([HumanMessage("Hi")]), handler
    )

    assert isinstance(response, ModelResponse)
    assert response.result[-1] == _final()
    assert len(handler.requests) == 3


@pytest.mark.parametrize("bound", [0, -1])
def test_rejects_a_bound_below_one(bound: int) -> None:
    with pytest.raises(ValueError, match="max_continuations"):
        AnthropicPauseTurnMiddleware(max_continuations=bound)


async def test_async_sends_paused_turn_back_unchanged() -> None:
    question = HumanMessage("What is the latest LangGraph release?")
    paused = _paused(1)
    handler = _ScriptedHandler([paused, _final()])

    response = await AnthropicPauseTurnMiddleware().awrap_model_call(
        _request([question]), handler.acall
    )

    assert [r.messages for r in handler.requests] == [[question], [question, paused]]
    assert isinstance(response, ModelResponse)
    assert response.result == [paused, _final()]


async def test_async_raises_once_the_bound_is_spent() -> None:
    turn = [_paused(1), _paused(2)]
    handler = _ScriptedHandler(turn)

    with pytest.raises(PauseTurnLimitExceededError) as error:
        await AnthropicPauseTurnMiddleware(max_continuations=1).awrap_model_call(
            _request([HumanMessage("Hi")]), handler.acall
        )

    assert error.value.max_continuations == 1
    assert error.value.messages == turn
    assert all(m is r for m, r in zip(error.value.messages, turn, strict=True))
    assert len(handler.requests) == 2


class _ScriptedChatModel(BaseChatModel):
    """Chat model that answers each call with the next scripted message."""

    replies: Iterator[AIMessage]
    calls: list[list[BaseMessage]]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=next(self.replies))])

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"


def test_agent_finishes_a_paused_turn() -> None:
    paused = _paused(1)
    model = _ScriptedChatModel(replies=iter([paused, _final()]), calls=[])
    agent = create_agent(model=model, middleware=[AnthropicPauseTurnMiddleware()])

    result = agent.invoke({"messages": [HumanMessage("Latest LangGraph release?")]})

    assert len(model.calls) == 2
    assert [type(m) for m in model.calls[1]] == [HumanMessage, AIMessage]
    assert model.calls[1][-1].content == paused.content
    assert result["messages"][-1].response_metadata["stop_reason"] == "end_turn"
