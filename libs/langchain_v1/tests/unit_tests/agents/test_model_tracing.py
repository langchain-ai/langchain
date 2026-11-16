"""Agent model classification must not change callback or streaming configuration."""

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableBinding, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langsmith import Client, get_tracing_context, tracing_context
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
)
from langchain.agents.middleware.types import InputAgentState, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@pytest.fixture
def client() -> MagicMock:
    mock = MagicMock(spec=Client)
    mock.tracing_queue = None
    return mock


class _CaptureMetadata(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    @override
    def on_chat_model_start(
        self,
        *_args: Any,
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.metadata.append(metadata or {})


class _HelperCalls(AgentMiddleware):
    def __init__(self) -> None:
        self.model = FakeToolCallingModel(name="helper")

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        self.model.invoke([HumanMessage("before")])
        response = handler(request)
        self.model.invoke([HumanMessage("after")])
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        await self.model.ainvoke([HumanMessage("before")])
        response = await handler(request)
        await self.model.ainvoke([HumanMessage("after")])
        return response


@pytest.mark.parametrize("use_async", [False, True])
async def test_model_classification_is_trace_only(client: MagicMock, *, use_async: bool) -> None:
    captured = _CaptureMetadata()

    @tool
    def inspect_config(runtime: ToolRuntime) -> str:
        """Inspect the tool's configuration and call another model."""
        assert "ls_agent_type" not in runtime.config.get("configurable", {})
        assert "ls_agent_type" not in runtime.config.get("metadata", {})
        assert "ls_agent_type" not in (get_tracing_context().get("metadata") or {})
        return str(FakeToolCallingModel(name="tool-helper").invoke("tool").content)

    agent = create_agent(
        FakeToolCallingModel(
            name="main",
            tool_calls=[[{"name": "inspect_config", "args": {}, "id": "call"}], []],
        ),
        tools=[inspect_config],
        middleware=[_HelperCalls()],
    )
    with tracing_context(client=client, enabled=True, metadata={"tenant": "example"}):
        before = get_tracing_context()
        config: RunnableConfig = {"callbacks": [captured]}
        inputs: InputAgentState = {"messages": [HumanMessage("hello")]}
        if use_async:
            chunks = [
                chunk async for chunk in agent.astream(inputs, config, stream_mode="messages")
            ]
        else:
            chunks = await asyncio.to_thread(
                lambda: list(agent.stream(inputs, config, stream_mode="messages"))
            )
        assert get_tracing_context() == before

    posts = [call.kwargs for call in client.create_run.call_args_list]
    main = [post for post in posts if post["name"] == "main"]
    assert len(main) == 2
    assert all(post["extra"]["metadata"]["ls_agent_type"] == "root" for post in main)
    assert all(post["extra"]["metadata"]["tenant"] == "example" for post in main)
    helpers = [post for post in posts if post["name"] in {"helper", "tool-helper"}]
    assert len(helpers) == 5
    assert all("ls_agent_type" not in post["extra"]["metadata"] for post in helpers)
    assert captured.metadata
    assert all("ls_agent_type" not in metadata for metadata in captured.metadata)
    assert chunks
    assert all("ls_agent_type" not in metadata for _, metadata in chunks)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("source", "agent_type"),
    [
        ("context", "subagent"),
        ("metadata", "middleware"),
        ("configurable", "subagent"),
        ("model", "compaction"),
        ("bound_metadata", "compaction"),
        ("bound_configurable", "subagent"),
        ("tracer", "subagent"),
        ("model_tracer", "compaction"),
        ("bound_tracer", "compaction"),
        ("bound_tracer_manager", "compaction"),
        ("manager_metadata", "compaction"),
        ("factory_metadata", "compaction"),
    ],
)
async def test_explicit_model_classification(
    client: MagicMock, *, use_async: bool, source: str, agent_type: str
) -> None:
    metadata = {"ls_agent_type": agent_type}
    config: RunnableConfig = {}
    model: BaseChatModel = FakeToolCallingModel(
        name="main", metadata=metadata if source == "model" else None
    )
    if source == "bound_metadata":
        model = cast("BaseChatModel", model.with_config(metadata=metadata))
    elif source == "bound_configurable":
        model = cast("BaseChatModel", model.with_config(configurable=metadata))
    elif source == "model_tracer":
        model.callbacks = [LangChainTracer(client=client, metadata=metadata)]
    elif source in {"bound_tracer", "bound_tracer_manager"}:
        tracer = LangChainTracer(client=client, metadata=metadata)
        callbacks = [tracer] if source == "bound_tracer" else CallbackManager([tracer])
        model = cast("BaseChatModel", model.with_config(callbacks=callbacks))
    elif source == "manager_metadata":
        model = cast(
            "BaseChatModel",
            model.with_config(
                callbacks=CallbackManager([LangChainTracer(client=client)], metadata=metadata)
            ),
        )
    elif source == "factory_metadata":

        def config_factory(_config: RunnableConfig) -> RunnableConfig:
            return {"metadata": metadata}

        model = cast(
            "BaseChatModel", RunnableBinding(bound=model, config_factories=[config_factory])
        )
    if source == "metadata":
        config["metadata"] = metadata
    elif source == "configurable":
        config["configurable"] = metadata
    elif source == "tracer":
        config["callbacks"] = [LangChainTracer(client=client, metadata=metadata)]
    agent = create_agent(model, middleware=[_HelperCalls()])
    with tracing_context(
        client=client, enabled=True, metadata=metadata if source == "context" else None
    ):
        if use_async:
            await agent.ainvoke({"messages": [HumanMessage("hello")]}, config)
        else:
            await asyncio.to_thread(
                lambda: agent.invoke({"messages": [HumanMessage("hello")]}, config)
            )

    main = [
        call.kwargs for call in client.create_run.call_args_list if call.kwargs["name"] == "main"
    ]
    assert len(main) == (2 if source == "model_tracer" else 1)
    assert all(post["extra"]["metadata"].get("ls_agent_type") != "root" for post in main)
    assert any(post["extra"]["metadata"].get("ls_agent_type") == agent_type for post in main)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("wrapper", ["binding_factory", "opaque"])
async def test_dynamic_model_wrappers_are_not_defaulted(
    client: MagicMock, *, use_async: bool, wrapper: str
) -> None:
    model = FakeToolCallingModel(name="main")
    factory_calls = 0

    def config_factory(_config: RunnableConfig) -> RunnableConfig:
        nonlocal factory_calls
        factory_calls += 1
        return {}

    wrapped = (
        RunnableBinding(bound=model, config_factories=[config_factory])
        if wrapper == "binding_factory"
        else RunnableLambda(model.invoke, afunc=model.ainvoke)
    )
    agent = create_agent(cast("BaseChatModel", wrapped))
    with tracing_context(client=client, enabled=True):
        inputs: InputAgentState = {"messages": [HumanMessage("hello")]}
        if use_async:
            await agent.ainvoke(inputs)
        else:
            await asyncio.to_thread(agent.invoke, inputs)

    main = [
        call.kwargs for call in client.create_run.call_args_list if call.kwargs["name"] == "main"
    ]
    assert len(main) == 1
    assert "ls_agent_type" not in main[0]["extra"]["metadata"]
    assert factory_calls == (1 if wrapper == "binding_factory" else 0)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("name", [None, "subagent"])
@pytest.mark.parametrize("agent_type", [None, "subagent"])
async def test_nested_agent_streams_preserve_classification(
    client: MagicMock, *, use_async: bool, name: str | None, agent_type: str | None
) -> None:
    class EmitCustom(AgentMiddleware):
        @override
        def before_model(self, _state: Any, runtime: Any) -> None:
            runtime.stream_writer("child-event")

    child = create_agent(
        FakeToolCallingModel(name="child", index=100), middleware=[EmitCustom()], name=name
    )

    @tool
    def delegate() -> str:
        """Delegate to a nested agent."""
        config: RunnableConfig = {}
        if agent_type:
            config["configurable"] = {"ls_agent_type": agent_type}
        return str(
            child.invoke({"messages": [HumanMessage("child")]}, config)["messages"][-1].content
        )

    parent = create_agent(
        FakeToolCallingModel(
            name="parent", tool_calls=[[{"name": "delegate", "args": {}, "id": "call"}], []]
        ),
        tools=[delegate],
    )
    with tracing_context(client=client, enabled=True):
        inputs: InputAgentState = {"messages": [HumanMessage("parent")]}
        if use_async:
            chunks = [
                chunk
                async for chunk in parent.astream(
                    inputs, stream_mode=["messages", "updates", "values", "custom"], subgraphs=True
                )
            ]
        else:
            chunks = await asyncio.to_thread(
                lambda: list(
                    parent.stream(
                        inputs,
                        stream_mode=["messages", "updates", "values", "custom"],
                        subgraphs=True,
                    )
                )
            )

    posts = [call.kwargs for call in client.create_run.call_args_list]
    parent_calls = [post for post in posts if post["name"] == "parent"]
    child_calls = [post for post in posts if post["name"] == "child"]
    assert len(parent_calls) == 2
    assert all(post["extra"]["metadata"]["ls_agent_type"] == "root" for post in parent_calls)
    assert len(child_calls) == 1
    assert child_calls[0]["extra"]["metadata"].get("ls_agent_type") == agent_type
    assert {mode for namespace, mode, _ in chunks if namespace} == {
        "messages",
        "updates",
        "values",
        "custom",
    }


@pytest.mark.parametrize("use_async", [False, True])
async def test_model_classification_survives_summarization(
    client: MagicMock, *, use_async: bool
) -> None:
    agent = create_agent(
        FakeToolCallingModel(name="main"),
        middleware=[
            SummarizationMiddleware(
                model=FakeToolCallingModel(name="summary"),
                trigger=("messages", 3),
                keep=("messages", 1),
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "summarization"}}
    with tracing_context(client=client, enabled=True):
        for turn in range(2):
            inputs: InputAgentState = {"messages": [HumanMessage(f"turn {turn}")]}
            if use_async:
                result = await agent.ainvoke(inputs, config)
            else:
                result = await asyncio.to_thread(agent.invoke, inputs, config)

    assert len(result["messages"]) == 3
    assert result["messages"][0].additional_kwargs["lc_source"] == "summarization"
    posts = [call.kwargs for call in client.create_run.call_args_list]
    main = [post for post in posts if post["name"] == "main"]
    summaries = [post for post in posts if post["name"] == "summary" and post["run_type"] == "llm"]
    assert len(main) == 2
    assert all(post["extra"]["metadata"]["ls_agent_type"] == "root" for post in main)
    assert len(summaries) == 1
    assert "ls_agent_type" not in summaries[0]["extra"]["metadata"]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
async def test_model_classification_context_restored_after_error(
    client: MagicMock, *, use_async: bool, fallback: bool
) -> None:
    class FailingModel(FakeToolCallingModel):
        @override
        def _generate(self, *_args: Any, **_kwargs: Any) -> Any:
            assert "ls_agent_type" not in get_config().get("metadata", {})
            msg = "model failed"
            raise ValueError(msg)

    middleware: list[AgentMiddleware] = [_HelperCalls()]
    if fallback:
        middleware.append(ModelFallbackMiddleware(FakeToolCallingModel(name="fallback")))
    agent = create_agent(FailingModel(name="failing"), middleware=middleware)
    with tracing_context(client=client, enabled=True, tags=["caller"]):
        before = get_tracing_context()
        inputs: InputAgentState = {"messages": [HumanMessage("hello")]}
        result = agent.ainvoke(inputs) if use_async else asyncio.to_thread(agent.invoke, inputs)
        if fallback:
            await result
        else:
            with pytest.raises(ValueError, match="model failed"):
                await result
        assert get_tracing_context() == before

    posts = [call.kwargs for call in client.create_run.call_args_list]
    model_calls = [post for post in posts if post["run_type"] == "llm" and post["name"] != "helper"]
    assert len(model_calls) == (2 if fallback else 1)
    assert all(post["extra"]["metadata"]["ls_agent_type"] == "root" for post in model_calls)
    assert all("caller" in post["tags"] for post in model_calls)
    assert all(
        "ls_agent_type" not in post["extra"]["metadata"]
        for post in posts
        if post["name"] == "helper"
    )


class _HandlerOnWorkerThread(AgentMiddleware):
    """Runs the handler on a thread that doesn't copy the runnable context."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(handler, request).result()

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        async def call() -> ModelResponse[Any]:
            return await handler(request)

        return await asyncio.get_running_loop().run_in_executor(None, asyncio.run, call())


def test_model_call_outside_runnable_context() -> None:
    agent = create_agent(model=FakeToolCallingModel(), middleware=[_HandlerOnWorkerThread()])

    result = agent.invoke({"messages": [HumanMessage("hi")]})

    assert result["messages"][-1].type == "ai"


async def test_async_model_call_outside_runnable_context() -> None:
    agent = create_agent(model=FakeToolCallingModel(), middleware=[_HandlerOnWorkerThread()])

    result = await agent.ainvoke({"messages": [HumanMessage("hi")]})

    assert result["messages"][-1].type == "ai"
