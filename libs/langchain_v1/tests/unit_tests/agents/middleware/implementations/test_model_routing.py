"""Tests for criteria-based model routing."""

import asyncio
from collections.abc import Sequence
from itertools import cycle
from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.agents.middleware import (
    InputAgentState,
    ModelRoutingInput,
    ModelRoutingMiddleware,
)
from langchain.agents.middleware.internal_call_transformer import internal_call_metadata
from tests.unit_tests.agents.model import FakeToolCallingModel

CRITERIA = {"fast": "Lookups and mechanical edits.", "reasoning": "Architectural tradeoffs."}


class CaptureChatInputs(BaseCallbackHandler):
    def __init__(self) -> None:
        self.inputs: list[list[BaseMessage]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del serialized, run_id, kwargs
        self.inputs.extend(messages)


def _models() -> dict[str, BaseChatModel]:
    return {
        name: GenericFakeChatModel(messages=cycle([AIMessage(content=name)])) for name in CRITERIA
    }


def _llm_selector(selection: dict[str, Any]) -> FakeToolCallingModel:
    return FakeToolCallingModel(
        tool_calls=[[{"name": "ModelRoutingResponse", "id": "route", "args": selection}]]
    )


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("backend", ["llm", "decision"])
async def test_routes_agent_with_either_backend(backend: str, *, use_async: bool) -> None:
    models = _models()
    capture = CaptureChatInputs()
    classifier_inputs: list[ModelRoutingInput] = []

    def classify(value: ModelRoutingInput) -> str:
        classifier_inputs.append(value)
        return "fast"

    middleware = ModelRoutingMiddleware(
        models=CRITERIA,
        model=_llm_selector({"model": "fast"}) if backend == "llm" else None,
        decision_model=RunnableLambda(classify) if backend == "decision" else None,
        system_prompt="Choose a safe profile.",
        model_factory=models.__getitem__,
    )
    agent = create_agent(models["reasoning"], middleware=[middleware], system_prompt="Agent policy")
    inputs: InputAgentState = {
        "messages": [HumanMessage("Earlier request"), HumanMessage("Current task")]
    }
    config: RunnableConfig = {"callbacks": [capture]}
    result = (
        await agent.ainvoke(inputs, config)
        if use_async
        else await asyncio.to_thread(agent.invoke, inputs, config)
    )

    assert result["messages"][-1].content == "fast"
    assert "model_route" not in result
    if backend == "llm":
        routing_messages = capture.inputs[0]
        assert len(routing_messages) == 2
        assert isinstance(routing_messages[0], SystemMessage)
        assert routing_messages[0].text.startswith("Choose a safe profile.")
        assert all(criterion in routing_messages[0].text for criterion in CRITERIA.values())
        assert routing_messages[1] == HumanMessage("Current task")
    else:
        assert classifier_inputs == [
            {
                "task": "Current task",
                "system_prompt": "Choose a safe profile.",
                "criteria": CRITERIA,
            }
        ]
    assert capture.inputs[-1][0] == SystemMessage("Agent policy")


@pytest.mark.parametrize("use_async", [False, True])
async def test_route_reused_in_tool_loop_and_reselected_on_next_turn(*, use_async: bool) -> None:
    selected_tasks: list[str] = []

    def classify(value: ModelRoutingInput) -> str:
        selected_tasks.append(value["task"])
        return "fast" if value["task"] == "simple" else "reasoning"

    @tool
    def lookup() -> str:
        """Return a lookup result."""
        return "found"

    fast = FakeToolCallingModel(tool_calls=[[{"name": "lookup", "id": "lookup", "args": {}}], []])
    reasoning = FakeToolCallingModel()
    models = {"fast": fast, "reasoning": reasoning}
    middleware = ModelRoutingMiddleware(
        models=CRITERIA,
        decision_model=RunnableLambda(classify),
        model_factory=models.__getitem__,
    )
    agent = create_agent(
        reasoning,
        tools=[lookup],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "routing"}}
    for task in ["simple", "complex"]:
        inputs: InputAgentState = {"messages": [HumanMessage(task)]}
        if use_async:
            await agent.ainvoke(inputs, config)
        else:
            await asyncio.to_thread(agent.invoke, inputs, config)

    assert selected_tasks == ["simple", "complex"]
    assert fast.index == 2
    assert reasoning.index == 1


@pytest.mark.parametrize("use_async", [False, True])
async def test_custom_task_extraction_and_selector_config(*, use_async: bool) -> None:
    received: list[tuple[ModelRoutingInput, RunnableConfig]] = []

    def classify(value: ModelRoutingInput, config: RunnableConfig) -> str:
        received.append((value, config))
        return "fast"

    def extract(messages: Sequence[BaseMessage]) -> str:
        return messages[0].text.removeprefix("envelope:")

    middleware = ModelRoutingMiddleware(
        models=CRITERIA,
        decision_model=RunnableLambda(classify),
        task_extractor=extract,
        model_factory=_models().__getitem__,
    )
    agent = create_agent(_models()["reasoning"], middleware=[middleware])
    inputs: InputAgentState = {
        "messages": [HumanMessage("envelope:task"), HumanMessage("injected metadata")]
    }
    config: RunnableConfig = {"tags": ["caller"], "metadata": {"request_id": "test"}}
    if use_async:
        await agent.ainvoke(inputs, config)
    else:
        await asyncio.to_thread(agent.invoke, inputs, config)
    value, selector_config = received[0]
    assert value["task"] == "task"
    assert "caller" in selector_config["tags"]
    assert "nostream" in selector_config["tags"]
    assert selector_config["metadata"]["request_id"] == "test"
    assert selector_config["metadata"].items() >= internal_call_metadata().items()


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("backend", ["llm", "decision"])
async def test_rejects_unknown_routes_before_model_resolution(
    backend: str, *, use_async: bool
) -> None:
    resolved: list[str] = []

    def factory(name: str) -> BaseChatModel:
        resolved.append(name)
        return _models()["fast"]

    middleware = ModelRoutingMiddleware(
        models=CRITERIA,
        model=_llm_selector({"model": "unconfigured:model"}) if backend == "llm" else None,
        decision_model=RunnableLambda(lambda _: "unconfigured:model")
        if backend == "decision"
        else None,
        model_factory=factory,
    )
    agent = create_agent(_models()["reasoning"], middleware=[middleware])
    inputs: InputAgentState = {"messages": [HumanMessage("Task")]}
    if use_async:
        with pytest.raises(ValueError, match="configured model names"):
            await agent.ainvoke(inputs)
    else:
        with pytest.raises(ValueError, match="configured model names"):
            await asyncio.to_thread(agent.invoke, inputs)
    assert not resolved


@pytest.mark.parametrize("selection", [{}, {"model": ["fast"]}, {"model": None}])
async def test_rejects_malformed_structured_output(selection: dict[str, Any]) -> None:
    middleware = ModelRoutingMiddleware(models=CRITERIA, model=_llm_selector(selection))
    with pytest.raises(ValueError, match="configured model names"):
        middleware.select_model("Task")
    with pytest.raises(ValueError, match="configured model names"):
        await middleware.aselect_model("Task")


def test_requires_models_and_exactly_one_backend() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        ModelRoutingMiddleware(models={}, model=_llm_selector({"model": "fast"}))
    with pytest.raises(ValueError, match="exactly one"):
        ModelRoutingMiddleware(models=CRITERIA)
    with pytest.raises(ValueError, match="exactly one"):
        ModelRoutingMiddleware(
            models=CRITERIA,
            model=_llm_selector({"model": "fast"}),
            decision_model=RunnableLambda(lambda _: "fast"),
        )


def test_missing_human_message_is_explicit_error() -> None:
    middleware = ModelRoutingMiddleware(
        models=CRITERIA, decision_model=RunnableLambda(lambda _: "fast")
    )
    agent = create_agent(_models()["reasoning"], middleware=[middleware])
    with pytest.raises(ValueError, match="human message"):
        agent.invoke({"messages": [AIMessage("No task")]})
