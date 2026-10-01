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
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
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

    def classify(inputs: ModelRoutingInput, config: RunnableConfig) -> str:
        assert inputs == {
            "messages": [original.messages[-1]],
            "system_prompt": "Custom routing instructions",
            "criteria": criteria,
        }
        assert config["metadata"].items() >= internal_call_metadata().items()
        selection_modes.append("sync")
        return "large"

    async def aclassify(inputs: ModelRoutingInput, config: RunnableConfig) -> str:
        result = classify(inputs, config)
        selection_modes[-1] = "async"
        return result

    middleware = ModelRoutingMiddleware(
        models=models,
        system_prompt="Custom routing instructions",
        routing_model=router if backend == "llm" else None,
        decision_model=RunnableLambda(classify, afunc=aclassify)
        if backend == "classifier"
        else None,
    )

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
    models: dict[str, ModelRoutingConfig] = {
        "large": {"model": FakeListChatModel(responses=["fallback"]), "criteria": "All tasks"}
    }
    router = RoutingChatModel(responses=[""], selection=selection)

    def classify(_inputs: ModelRoutingInput) -> Any:
        return selection

    middleware = ModelRoutingMiddleware(
        models=models,
        routing_model=router if backend == "llm" else None,
        decision_model=RunnableLambda(classify) if backend == "classifier" else None,
    )
    with pytest.raises(ValueError, match="configured route name"):
        middleware.before_model(ModelRoutingState(messages=make_request().messages), Runtime())
    with pytest.raises(ValueError, match="configured route name"):
        await middleware.abefore_model(
            ModelRoutingState(messages=make_request().messages), Runtime()
        )
    middleware.fallback_route = "large"
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
    response = await middleware.awrap_model_call(original, ahandler)
    assert response.result[0].content == "fallback"


async def test_custom_input_and_no_cross_request_cache() -> None:
    def extract(state: ModelRoutingState) -> Sequence[BaseMessage]:
        return [message for message in state["messages"] if message.text != "Injected context"]

    def classify(inputs: ModelRoutingInput) -> str:
        return "small" if inputs["messages"][-1].text == "Lookup" else "large"

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
        ({"decision_model": None}, "exactly one"),
        ({"routing_model": FakeListChatModel(responses=[""])}, "exactly one"),
        ({"fallback_route": "unknown"}, "fallback_route"),
    ],
)
def test_configuration_validation(kwargs: dict[str, Any], match: str) -> None:
    options: dict[str, Any] = {
        "models": {
            "small": {"model": FakeListChatModel(responses=["small"]), "criteria": "All tasks"}
        },
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


async def test_agent_uses_selected_model() -> None:
    middleware = ModelRoutingMiddleware(
        models={
            "small": {
                "model": FakeListChatModel(responses=["Selected response"]),
                "criteria": "All tasks",
            }
        },
        decision_model=RunnableLambda(lambda _: "small"),
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
        decision_model=RunnableLambda(lambda _: next(selections)),
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
