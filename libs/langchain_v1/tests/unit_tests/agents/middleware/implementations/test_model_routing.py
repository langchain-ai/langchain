"""Behavioral tests for model routing middleware."""

import importlib.util
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core._api import LangChainBetaWarning
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.runtime import Runtime
from pydantic import Field
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import (
    InputAgentState,
    ModelRequest,
    ModelResponse,
    ModelRoutingMiddleware,
    model_routing,
)
from langchain.agents.middleware.internal_call_transformer import internal_call_metadata
from langchain.agents.middleware.model_routing import (
    ModelRoutingConfig,
    ModelRoutingInput,
    ModelRoutingOutput,
    ModelRoutingState,
)


def test_beta_warning() -> None:
    spec = importlib.util.spec_from_file_location("isolated_model_routing", model_routing.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.warns(LangChainBetaWarning, match="may change or be removed without notice"):
        module.ModelRoutingMiddleware(
            models={
                "small": {"model": FakeListChatModel(responses=["small"]), "criteria": "All tasks"}
            },
            decision_model=RunnableLambda(lambda _: ModelRoutingOutput(route="small")),
        )


class RoutingChatModel(FakeListChatModel):
    response: object = Field(default_factory=lambda: {"route": "large"})
    routing_schema: dict[str, Any] | None = None
    received_messages: list[BaseMessage] = Field(default_factory=list)
    selection_modes: list[str] = Field(default_factory=list)

    @override
    def with_structured_output(
        self, schema: dict[str, Any] | type, **_kwargs: Any
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
            self.selection_modes.append("sync")
            return self.response

        async def aselect(messages: LanguageModelInput, config: RunnableConfig) -> object:
            response = select(messages, config)
            self.selection_modes[-1] = "async"
            return response

        return RunnableLambda(select, afunc=aselect)


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


@pytest.mark.parametrize("backend", ["llm", "llm_string", "classifier"])
@pytest.mark.parametrize("async_mode", [False, True])
async def test_route_and_preserve_request(backend: str, *, async_mode: bool) -> None:
    original = make_request()
    models: dict[str, ModelRoutingConfig] = {
        "small": {
            "model": FakeListChatModel(responses=["small response"]),
            "criteria": "Simple tasks",
        },
        "large": {
            "model": FakeListChatModel(responses=["large response"]),
            "criteria": "Complex tasks",
        },
    }
    criteria = {"small": "Simple tasks", "large": "Complex tasks"}
    selection_modes: list[str] = []
    router = RoutingChatModel(responses=[""])

    def classify(inputs: ModelRoutingInput, config: RunnableConfig) -> ModelRoutingOutput:
        assert inputs == {
            "messages": [original.messages[-1]],
            "instructions": "Custom routing instructions",
            "criteria": criteria,
        }
        assert config["metadata"].items() >= internal_call_metadata().items()
        selection_modes.append("sync")
        return {"route": "large"}

    async def aclassify(inputs: ModelRoutingInput, config: RunnableConfig) -> ModelRoutingOutput:
        result = classify(inputs, config)
        selection_modes[-1] = "async"
        return result

    decision_model: str | RoutingChatModel | Runnable[ModelRoutingInput, ModelRoutingOutput]
    if backend == "llm_string":
        decision_model = "test:router"
    elif backend == "llm":
        decision_model = router
    else:
        decision_model = RunnableLambda(classify, afunc=aclassify)
    with patch(
        "langchain.agents.middleware.model_routing.init_chat_model", return_value=router
    ) as init_model:
        middleware = ModelRoutingMiddleware(
            models=models,
            instructions="Custom routing instructions",
            decision_model=decision_model,
        )
    if backend == "llm_string":
        init_model.assert_called_once_with("test:router")
    else:
        init_model.assert_not_called()

    def handler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[request.model.invoke(request.messages)])

    async def ahandler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[await request.model.ainvoke(request.messages)])

    state = ModelRoutingState(messages=original.messages)
    update = (
        await middleware.abefore_model(state, Runtime())
        if async_mode
        else middleware.before_model(state, Runtime())
    )
    state["model_route"] = update["model_route"]
    original = original.override(state=state)
    response = (
        await middleware.awrap_model_call(original, ahandler)
        if async_mode
        else middleware.wrap_model_call(original, handler)
    )
    assert response.result[0].content == "large response"
    if backend == "classifier":
        assert selection_modes == ["async" if async_mode else "sync"]
    if backend != "classifier":
        assert router.selection_modes == ["async" if async_mode else "sync"]
        assert router.routing_schema is not None
        assert router.routing_schema["type"] == "object"
        assert router.routing_schema["required"] == ["route"]
        assert router.routing_schema["additionalProperties"] is False
        assert router.routing_schema["properties"]["route"]["type"] == "string"
        assert router.routing_schema["properties"]["route"]["enum"] == ["small", "large"]
        assert router.received_messages[-1] == original.messages[-1]
        assert "Custom routing instructions" in router.received_messages[0].text
        assert "Simple tasks" in router.received_messages[0].text
        assert "Complex tasks" in router.received_messages[0].text


def test_routing_schema_preserves_each_instances_route_names() -> None:
    route_sets = [["fast-model", "provider:model"], ["_ignore_", "__members__"]]
    routers = [
        RoutingChatModel(responses=[""], response={"route": routes[0]}) for routes in route_sets
    ]
    for routes, router in zip(route_sets, routers, strict=True):
        middleware = ModelRoutingMiddleware(
            models={
                route: {"model": FakeListChatModel(responses=[route]), "criteria": route}
                for route in routes
            },
            decision_model=router,
        )
        state = ModelRoutingState(messages=[HumanMessage(content="Task")])
        assert middleware.before_model(state, Runtime()) == {"model_route": routes[0]}

    for routes, router in zip(route_sets, routers, strict=True):
        assert router.routing_schema is not None
        assert router.routing_schema["properties"]["route"]["enum"] == routes


@pytest.mark.parametrize("backend", ["llm", "classifier"])
@pytest.mark.parametrize(
    "response",
    [{"route": "unknown"}, {"route": None}, {"route": ["large"]}, {}, None, "large", ["large"]],
)
async def test_invalid_selection_and_explicit_fallback(backend: str, response: object) -> None:
    models: dict[str, ModelRoutingConfig] = {
        "large": {"model": FakeListChatModel(responses=["fallback"]), "criteria": "All tasks"}
    }
    router = RoutingChatModel(responses=[""], response=response)

    def classify(_inputs: ModelRoutingInput) -> Any:
        return response

    middleware = ModelRoutingMiddleware(
        models=models,
        decision_model=RunnableLambda(classify) if backend == "classifier" else router,
    )
    with pytest.raises(ValueError, match="configured route name"):
        middleware.before_model(ModelRoutingState(messages=make_request().messages), Runtime())
    with pytest.raises(ValueError, match="configured route name"):
        await middleware.abefore_model(
            ModelRoutingState(messages=make_request().messages), Runtime()
        )
    middleware.fallback_route = "large"
    assert await middleware.abefore_model(
        ModelRoutingState(messages=make_request().messages), Runtime()
    ) == {"model_route": "large"}
    original = make_request()
    original = original.override(
        state=ModelRoutingState(
            messages=original.messages,
            model_route=middleware.before_model(
                ModelRoutingState(messages=original.messages), Runtime()
            )["model_route"],
        )
    )

    def handler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[request.model.invoke(request.messages)])

    async def ahandler(request: ModelRequest) -> ModelResponse:
        check_request(request, original)
        return ModelResponse(result=[await request.model.ainvoke(request.messages)])

    assert middleware.wrap_model_call(original, handler).result[0].content == "fallback"
    model_response = await middleware.awrap_model_call(original, ahandler)
    assert model_response.result[0].content == "fallback"


async def test_custom_input_and_no_cross_request_cache() -> None:
    def extract(state: ModelRoutingState) -> Sequence[BaseMessage]:
        return [message for message in state["messages"] if message.text != "Injected context"]

    def classify(inputs: ModelRoutingInput) -> ModelRoutingOutput:
        return {"route": "small" if inputs["messages"][-1].text == "Lookup" else "large"}

    middleware = ModelRoutingMiddleware(
        models={
            "small": {"model": FakeListChatModel(responses=["small"]), "criteria": "Lookup"},
            "large": {"model": FakeListChatModel(responses=["large"]), "criteria": "Reasoning"},
        },
        decision_model=RunnableLambda(classify),
        input_extractor=extract,
    )
    first = make_request().override(
        messages=[HumanMessage(content="Lookup"), HumanMessage(content="Injected context")]
    )
    second = make_request()
    assert (
        middleware.before_model(ModelRoutingState(messages=first.messages), Runtime())[
            "model_route"
        ]
        == "small"
    )
    assert (await middleware.abefore_model(ModelRoutingState(messages=second.messages), Runtime()))[
        "model_route"
    ] == "large"
    assert (await middleware.abefore_model(ModelRoutingState(messages=first.messages), Runtime()))[
        "model_route"
    ] == "small"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"models": {}}, "non-empty"),
        ({"fallback_route": "unknown"}, "fallback_route"),
    ],
)
def test_configuration_validation(kwargs: dict[str, Any], match: str) -> None:
    options: dict[str, Any] = {
        "models": {
            "small": {"model": FakeListChatModel(responses=["small"]), "criteria": "All tasks"}
        },
        "decision_model": RunnableLambda(lambda _: ModelRoutingOutput(route="small")),
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ModelRoutingMiddleware(**options)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "required keyword-only argument: 'decision_model'"),
        (
            {"routing_model": FakeListChatModel(responses=[""])},
            "unexpected keyword argument 'routing_model'",
        ),
    ],
)
def test_decision_model_required(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(TypeError, match=match):
        ModelRoutingMiddleware(
            models={
                "small": {"model": FakeListChatModel(responses=["small"]), "criteria": "All tasks"}
            },
            **kwargs,
        )


async def test_backend_errors_propagate_and_missing_input_is_explicit() -> None:
    def classify(_inputs: ModelRoutingInput) -> ModelRoutingOutput:
        msg = "Classifier unavailable"
        raise RuntimeError(msg)

    middleware = ModelRoutingMiddleware(
        models={
            "small": {"model": FakeListChatModel(responses=["small"]), "criteria": "All tasks"}
        },
        decision_model=RunnableLambda(classify),
        fallback_route="small",
    )
    with pytest.raises(RuntimeError, match="Classifier unavailable"):
        middleware.before_model(ModelRoutingState(messages=make_request().messages), Runtime())
    with pytest.raises(RuntimeError, match="Classifier unavailable"):
        await middleware.abefore_model(
            ModelRoutingState(messages=make_request().messages), Runtime()
        )
    with pytest.raises(ValueError, match="No routing messages"):
        middleware.before_model(
            ModelRoutingState(messages=[AIMessage(content="No user")]), Runtime()
        )


@pytest.mark.parametrize("backend", ["llm", "classifier"])
async def test_agent_uses_selected_model(backend: str) -> None:
    middleware = ModelRoutingMiddleware(
        models={
            "small": {
                "model": FakeListChatModel(responses=["Selected response"]),
                "criteria": "All tasks",
            }
        },
        decision_model=RoutingChatModel(responses=[""], response={"route": "small"})
        if backend == "llm"
        else RunnableLambda(lambda _: ModelRoutingOutput(route="small")),
    )
    agent = create_agent(
        model=FakeListChatModel(responses=["Original response"]),
        middleware=[middleware],
    )
    inputs: InputAgentState = {"messages": [HumanMessage(content="Do this task")]}
    assert agent.invoke(inputs)["messages"][-1].content == "Selected response"
    assert (await agent.ainvoke(inputs))["messages"][-1].content == "Selected response"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_route_persists_until_cleared(*, async_mode: bool) -> None:
    selections = iter(["small", "large"])
    middleware = ModelRoutingMiddleware(
        models={
            "small": {"model": FakeListChatModel(responses=["small"]), "criteria": "Lookup"},
            "large": {"model": FakeListChatModel(responses=["large"]), "criteria": "Reasoning"},
        },
        decision_model=RunnableLambda(lambda _: ModelRoutingOutput(route=next(selections))),
    )
    state = ModelRoutingState(messages=[HumanMessage(content="Lookup")])

    async def prepare() -> str:
        if async_mode:
            return (await middleware.abefore_model(state, Runtime()))["model_route"]
        return middleware.before_model(state, Runtime())["model_route"]

    state["model_route"] = await prepare()
    assert state["model_route"] == "small"
    state["messages"] = [HumanMessage(content="Reasoning")]
    state["model_route"] = await prepare()
    assert state["model_route"] == "small"
    state["model_route"] = None
    state["model_route"] = await prepare()
    assert state["model_route"] == "large"
