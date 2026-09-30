"""Behavioral tests for model routing middleware."""

import importlib.util
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core._api import LangChainBetaWarning
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel, Field
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import (
    InputAgentState,
    ModelRequest,
    ModelResponse,
    ModelRoutingInput,
    ModelRoutingMiddleware,
    model_routing,
)
from langchain.agents.middleware.internal_call_transformer import internal_call_metadata


def test_beta_warning() -> None:
    spec = importlib.util.spec_from_file_location("isolated_model_routing", model_routing.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.warns(LangChainBetaWarning, match="may change or be removed without notice"):
        module.ModelRoutingMiddleware(
            models={"small": FakeListChatModel(responses=["small"])},
            criteria={"small": "All tasks"},
            decision_model=RunnableLambda(lambda _: "small"),
        )


class RoutingChatModel(FakeListChatModel):
    selection: object = "large"
    routing_schema: dict[str, Any] | None = None
    received_messages: list[BaseMessage] = Field(default_factory=list)

    @override
    def with_structured_output(
        self, schema: dict[str, Any] | type[BaseModel], **_kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        assert isinstance(schema, dict)
        self.routing_schema = schema

        def select(messages: LanguageModelInput, config: RunnableConfig) -> object:
            assert isinstance(messages, list)
            assert all(isinstance(message, BaseMessage) for message in messages)
            self.received_messages = [
                message for message in messages if isinstance(message, BaseMessage)
            ]
            assert config["metadata"].items() >= internal_call_metadata().items()
            return {"route": self.selection}

        return RunnableLambda(select)


def make_request() -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["original"]),
        messages=[HumanMessage(content="Earlier task"), HumanMessage(content="Current task")],
        system_message=SystemMessage(content="Agent instructions"),
        tools=[{"type": "custom"}],
        tool_choice="auto",
        model_settings={"temperature": 0.2},
    )


def check_request(request: ModelRequest, original: ModelRequest) -> None:
    assert request is not original
    assert request.messages is original.messages
    assert request.system_message is original.system_message
    assert request.tools is original.tools
    assert request.tool_choice == original.tool_choice
    assert request.model_settings is original.model_settings
    assert request.response_format is original.response_format
    assert request.state is original.state
    assert request.runtime is original.runtime
    assert original.model.invoke("task").content == "original"


@pytest.mark.parametrize("backend", ["llm", "classifier"])
@pytest.mark.parametrize("async_mode", [False, True])
async def test_route_and_preserve_request(backend: str, *, async_mode: bool) -> None:
    original = make_request()
    models = {
        "small": FakeListChatModel(responses=["small response"]),
        "large": FakeListChatModel(responses=["large response"]),
    }
    criteria = {"small": "Simple tasks", "large": "Complex tasks"}
    router = RoutingChatModel(responses=[""])

    def classify(inputs: ModelRoutingInput, config: RunnableConfig) -> str:
        assert inputs == {
            "messages": [original.messages[-1]],
            "system_prompt": "Custom routing instructions",
            "criteria": criteria,
        }
        assert config["metadata"].items() >= internal_call_metadata().items()
        return "large"

    middleware = ModelRoutingMiddleware(
        models=models,
        criteria=criteria,
        system_prompt="Custom routing instructions",
        routing_model=router if backend == "llm" else None,
        decision_model=RunnableLambda(classify) if backend == "classifier" else None,
    )

    def handler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[request.model.invoke(request.messages)])

    async def ahandler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[await request.model.ainvoke(request.messages)])

    response = (
        await middleware.awrap_model_call(original, ahandler)
        if async_mode
        else middleware.wrap_model_call(original, handler)
    )
    assert response.result[0].content == "large response"
    if backend == "llm":
        assert router.routing_schema is not None
        assert router.routing_schema["properties"]["route"]["enum"] == ["small", "large"]
        assert router.received_messages[-1] == original.messages[-1]
        assert "Custom routing instructions" in router.received_messages[0].text
        assert "Simple tasks" in router.received_messages[0].text
        assert "Complex tasks" in router.received_messages[0].text


@pytest.mark.parametrize("backend", ["llm", "classifier"])
@pytest.mark.parametrize("selection", ["unknown", None, ["large"]])
async def test_invalid_selection_and_explicit_fallback(backend: str, selection: object) -> None:
    models = {"large": FakeListChatModel(responses=["fallback"])}
    router = RoutingChatModel(responses=[""], selection=selection)

    def classify(_inputs: ModelRoutingInput) -> Any:
        return selection

    middleware = ModelRoutingMiddleware(
        models=models,
        criteria={"large": "All tasks"},
        routing_model=router if backend == "llm" else None,
        decision_model=RunnableLambda(classify) if backend == "classifier" else None,
    )
    with pytest.raises(ValueError, match="configured route name"):
        middleware.select_route(make_request())
    with pytest.raises(ValueError, match="configured route name"):
        await middleware.aselect_route(make_request())
    middleware.fallback_route = "large"
    original = make_request()

    def handler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[request.model.invoke(request.messages)])

    async def ahandler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[await request.model.ainvoke(request.messages)])

    assert middleware.wrap_model_call(original, handler).result[0].content == "fallback"
    response = await middleware.awrap_model_call(original, ahandler)
    assert response.result[0].content == "fallback"


async def test_custom_input_and_no_cross_request_cache() -> None:
    def extract(request: ModelRequest) -> Sequence[BaseMessage]:
        return [message for message in request.messages if message.text != "Injected context"]

    def classify(inputs: ModelRoutingInput) -> str:
        return "small" if inputs["messages"][-1].text == "Lookup" else "large"

    middleware = ModelRoutingMiddleware(
        models={
            "small": FakeListChatModel(responses=["small"]),
            "large": FakeListChatModel(responses=["large"]),
        },
        criteria={"small": "Lookup", "large": "Reasoning"},
        decision_model=RunnableLambda(classify),
        input_extractor=extract,
    )
    first = make_request().override(
        messages=[HumanMessage(content="Lookup"), HumanMessage(content="Injected context")]
    )
    second = make_request()
    assert middleware.select_route(first) == "small"
    assert await middleware.aselect_route(second) == "large"
    assert await middleware.aselect_route(first) == "small"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"models": {}}, "non-empty"),
        ({"criteria": {}}, "same route names"),
        ({"decision_model": None}, "exactly one"),
        ({"routing_model": FakeListChatModel(responses=[""])}, "exactly one"),
        ({"fallback_route": "unknown"}, "fallback_route"),
    ],
)
def test_configuration_validation(kwargs: dict[str, Any], match: str) -> None:
    options: dict[str, Any] = {
        "models": {"small": FakeListChatModel(responses=["small"])},
        "criteria": {"small": "All tasks"},
        "decision_model": RunnableLambda(lambda _: "small"),
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ModelRoutingMiddleware(**options)


async def test_backend_errors_propagate_and_missing_input_is_explicit() -> None:
    def classify(_inputs: ModelRoutingInput) -> str:
        msg = "Classifier unavailable"
        raise RuntimeError(msg)

    middleware = ModelRoutingMiddleware(
        models={"small": FakeListChatModel(responses=["small"])},
        criteria={"small": "All tasks"},
        decision_model=RunnableLambda(classify),
        fallback_route="small",
    )
    with pytest.raises(RuntimeError, match="Classifier unavailable"):
        middleware.select_route(make_request())
    with pytest.raises(RuntimeError, match="Classifier unavailable"):
        await middleware.aselect_route(make_request())
    with pytest.raises(ValueError, match="No routing messages"):
        middleware.select_route(make_request().override(messages=[AIMessage(content="No user")]))


async def test_agent_uses_selected_model() -> None:
    middleware = ModelRoutingMiddleware(
        models={"small": FakeListChatModel(responses=["Selected response"])},
        criteria={"small": "All tasks"},
        decision_model=RunnableLambda(lambda _: "small"),
    )
    agent = create_agent(
        model=FakeListChatModel(responses=["Original response"]),
        middleware=[middleware],
    )
    inputs: InputAgentState = {"messages": [HumanMessage(content="Do this task")]}
    assert agent.invoke(inputs)["messages"][-1].content == "Selected response"
    assert (await agent.ainvoke(inputs))["messages"][-1].content == "Selected response"
