"""Tests for `OpenAIModelRouterMiddleware`."""

from __future__ import annotations

import itertools
import json
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState, omit_payload
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain_openai._compat import httpx
from langchain_openai.chat_models.base import OpenAIAPIError
from langchain_openai.decisions import ChoiceAnswer, OpenAIDecisions
from langchain_openai.middleware import ModelChoice, OpenAIModelRouterMiddleware

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
)


def _fake(name: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(name)]))


def _choice(route: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "name": "model_route",
        "choice": route,
        "probabilities": [
            {"value": "fast", "probability": 0.9 if route == "fast" else 0.1},
            {"value": "powerful", "probability": 0.1 if route == "fast" else 0.9},
        ],
        "confidence": 0.8,
    }


def _router(
    *answers: dict[str, Any],
    status_code: int = 200,
    observed: list[dict[str, Any]] | None = None,
) -> OpenAIModelRouterMiddleware:
    replies = iter(answers)

    def respond(request: httpx.Request) -> httpx.Response:
        if observed is not None:
            observed.append(json.loads(request.content))
        if status_code != 200:
            return httpx.Response(status_code, json={"error": {"message": "down"}})
        return httpx.Response(
            200, json={"model": "gpt-6-luna", "answers": [next(replies)]}
        )

    async def arespond(request: httpx.Request) -> httpx.Response:
        return respond(request)

    decisions = OpenAIDecisions(
        model="gpt-6-luna",
        api_key=SecretStr("test-api-key"),
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(arespond)),
    )
    return OpenAIModelRouterMiddleware(
        choices={
            "fast": ModelChoice(model=_fake("fast"), criteria="Simple tasks."),
            "powerful": ModelChoice(model=_fake("powerful"), criteria="Hard tasks."),
        },
        instructions="Choose the least costly model suited to the task.",
        model=decisions,
    )


async def _run(
    middleware: OpenAIModelRouterMiddleware,
    *,
    async_: bool,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    agent = create_agent(_fake("default"), middleware=[middleware])
    state = InputAgentState(messages=messages or [HumanMessage("Prove P != NP.")])
    return await agent.ainvoke(state) if async_ else agent.invoke(state)


@pytest.mark.parametrize("async_", [False, True])
async def test_agent_routes_using_latest_human_message(*, async_: bool) -> None:
    observed: list[dict[str, Any]] = []
    middleware = _router(_choice("powerful"), observed=observed)

    result = await _run(
        middleware,
        async_=async_,
        messages=[
            HumanMessage("Hi"),
            AIMessage("Hello!"),
            HumanMessage("Prove P != NP."),
        ],
    )

    assert result["messages"][-1].content == "powerful"
    assert isinstance(result["model_route"], ChoiceAnswer)
    assert result["model_route"].probabilities == {"fast": 0.1, "powerful": 0.9}
    [request] = observed
    assert request["input"] == [{"role": "user", "content": "Prove P != NP."}]
    assert request["questions"] == [
        {
            "type": "choice",
            "name": "model_route",
            "instructions": "Choose the least costly model suited to the task.",
            "choices": [
                {"value": "fast", "description": "Simple tasks."},
                {"value": "powerful", "description": "Hard tasks."},
            ],
        }
    ]


@pytest.mark.parametrize("async_", [False, True])
async def test_refusal_uses_agent_model(*, async_: bool) -> None:
    middleware = _router({"type": "refusal", "name": "model_route"})

    result = await _run(middleware, async_=async_)

    assert result["messages"][-1].content == "default"
    assert result["model_route"] is None


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "url": "https://example.com/a.png"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
        {
            "type": "file",
            "base64": "JVBERi0=",
            "mime_type": "application/pdf",
            "filename": "a.pdf",
        },
        {"type": "audio", "base64": "UklGRg==", "mime_type": "audio/wav"},
    ],
)
async def test_attachments_are_replaced_with_placeholders(
    block: dict[str, Any],
) -> None:
    observed: list[dict[str, Any]] = []
    middleware = _router(_choice("powerful"), observed=observed)
    message = HumanMessage(content=[{"type": "text", "text": "Review this."}, block])

    result = await _run(middleware, async_=False, messages=[message])

    assert result["messages"][-1].content == "powerful"
    assert observed[0]["input"] == [
        {"role": "user", "content": f"Review this.\n[{block['type']} omitted]"}
    ]


async def test_missing_human_message_skips_classification() -> None:
    observed: list[dict[str, Any]] = []
    middleware = _router(observed=observed)

    result = await _run(middleware, async_=False, messages=[AIMessage("Hello!")])

    assert result["messages"][-1].content == "default"
    assert result["model_route"] is None
    assert observed == []


def test_refusal_clears_route_from_previous_run() -> None:
    middleware = _router(
        _choice("powerful"), {"type": "refusal", "name": "model_route"}
    )
    agent = create_agent(
        _fake("default"), middleware=[middleware], checkpointer=InMemorySaver()
    )
    config: Any = {"configurable": {"thread_id": "thread"}}

    first = agent.invoke(InputAgentState(messages=[HumanMessage("Hard")]), config)
    second = agent.invoke(InputAgentState(messages=[HumanMessage("Easy")]), config)

    assert first["messages"][-1].content == "powerful"
    assert second["messages"][-1].content == "default"
    assert second["model_route"] is None


@pytest.mark.parametrize("async_", [False, True])
async def test_api_failure_terminates_agent_run(*, async_: bool) -> None:
    with pytest.raises(OpenAIAPIError):
        await _run(_router(status_code=500), async_=async_)


def test_model_strings_are_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    middleware = OpenAIModelRouterMiddleware(
        choices={"fast": ModelChoice(model="openai:gpt-5.4-mini", criteria="Simple.")},
        instructions="Choose a model.",
        model="gpt-6-luna",
    )

    assert isinstance(middleware.models["fast"], ChatOpenAI)
    assert middleware.decisions.model == "gpt-6-luna"


@pytest.mark.parametrize(
    ("choices", "instructions"),
    [({}, "Choose a model."), ({"fast": ModelChoice(model="x", criteria="y")}, " ")],
)
def test_invalid_configuration_is_rejected(
    choices: dict[str, ModelChoice], instructions: str
) -> None:
    with pytest.raises(ValueError, match="must"):
        OpenAIModelRouterMiddleware(
            choices=choices, instructions=instructions, model="gpt-6-luna"
        )


def test_is_beta() -> None:
    assert (OpenAIModelRouterMiddleware.__doc__ or "").startswith(".. beta::")


def test_trace_policy_omits_user_messages() -> None:
    assert _router().trace_policy.process_inputs is omit_payload
