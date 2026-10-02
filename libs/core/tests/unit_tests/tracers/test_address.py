"""Destination resolution for LangSmith tracing."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from itertools import product
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langsmith import Client, address, configure, tracing_context
from langsmith.address import EnvAddressError
from langsmith.run_trees import RunTree, WriteReplica
from langsmith.utils import LangSmithUserError, get_env_var, get_tracer_project

from langchain_core.callbacks.manager import (
    CallbackManager,
    atrace_as_chain_group,
    trace_as_chain_group,
)
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import (
    _get_trace_callbacks,
    _get_tracer_kwargs,
    tracing_v2_enabled,
)
from langchain_core.tracers.langchain import LangChainTracer

ADDRESS = address.agent("code-agent", "production")
ENV_ADDRESS = address.agent("env-agent", "staging")


@pytest.fixture(autouse=True)
def clean_destination(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "HOSTED_LANGSERVE_PROJECT_NAME",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_SESSION",
        "LANGSMITH_AGENT_ID",
        "LANGSMITH_AGENT_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()
    yield
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()


@pytest.mark.parametrize(
    ("code_project", "code_address", "env_project", "env_address"),
    list(product((False, True), repeat=4)),
)
def test_destination_matrix(
    monkeypatch: pytest.MonkeyPatch,
    *,
    code_project: bool,
    code_address: bool,
    env_project: bool,
    env_address: bool,
) -> None:
    if env_project:
        monkeypatch.setenv("LANGSMITH_PROJECT", "env-project")
    if env_address:
        monkeypatch.setenv("LANGSMITH_AGENT_ID", "env-agent")
        monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "staging")
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()
    client = MagicMock(spec=Client)
    kwargs: dict[str, Any] = {
        "client": client,
        "project_name": "code-project" if code_project else None,
        "address": ADDRESS if code_address else None,
    }
    tracer = LangChainTracer(**kwargs)
    run_id = uuid4()
    if code_project and code_address:
        with pytest.raises(LangSmithUserError, match="not both"):
            tracer.on_chain_start({}, {}, run_id=run_id)
        client.create_run.assert_not_called()
        client.update_run.assert_not_called()
        return
    invalid_env = not (code_project or code_address) and env_project and env_address
    if invalid_env:
        with pytest.raises(EnvAddressError):
            tracer.on_chain_start({}, {}, run_id=run_id)
        client.create_run.assert_not_called()
        client.update_run.assert_not_called()
        return
    tracer.on_chain_start({}, {"input": "hello"}, run_id=run_id)
    tracer.on_chain_end({}, run_id=run_id)
    expected_address = (
        ADDRESS
        if code_address
        else (ENV_ADDRESS if env_address and not code_project else None)
    )
    expected_project = (
        None
        if expected_address
        else (
            "code-project"
            if code_project
            else "env-project"
            if env_project
            else "default"
        )
    )
    client.create_run.assert_called_once()
    client.update_run.assert_called_once()
    for payload in (
        client.create_run.call_args.kwargs,
        client.update_run.call_args.kwargs,
    ):
        assert payload.get("session_name") == expected_project
        assert payload.get("address") == expected_address


@pytest.mark.parametrize("run_type", ["llm", "chat", "tool", "retriever"])
def test_callback_address_propagation(run_type: str) -> None:
    client = MagicMock(spec=Client)
    tracer = LangChainTracer(address=ADDRESS, client=client)
    run_id = uuid4()
    if run_type == "llm":
        tracer.on_llm_start({}, ["hello"], run_id=run_id)
        tracer.on_llm_end(LLMResult(generations=[]), run_id=run_id)
    elif run_type == "chat":
        tracer.on_chat_model_start({}, [[HumanMessage("hello")]], run_id=run_id)
        tracer.on_llm_end(LLMResult(generations=[]), run_id=run_id)
    elif run_type == "tool":
        tracer.on_tool_start({}, "hello", run_id=run_id)
        tracer.on_tool_end("ok", run_id=run_id)
    else:
        tracer.on_retriever_start({}, "hello", run_id=run_id)
        tracer.on_retriever_end([], run_id=run_id)
    client.create_run.assert_called_once()
    client.update_run.assert_called_once()
    for payload in (
        client.create_run.call_args.kwargs,
        client.update_run.call_args.kwargs,
    ):
        assert payload.get("address") == ADDRESS
        assert payload.get("session_name") is None


@pytest.mark.parametrize(
    "destination", [{"address": ADDRESS}, {"project_name": "configured"}]
)
@pytest.mark.parametrize("mode", ["context", "configure", "parent"])
def test_ambient_destination(destination: dict[str, str], mode: str) -> None:
    client = MagicMock(spec=Client)
    scope: AbstractContextManager[None]
    if mode == "context":
        scope = tracing_context(
            enabled=True,
            client=client,
            project_name=destination.get("project_name"),
            address=destination.get("address"),
        )
    elif mode == "configure":
        configure(
            enabled=True,
            client=client,
            project_name=destination.get("project_name"),
            address=destination.get("address"),
        )
        scope = nullcontext()
    else:
        parent = RunTree(name="parent", ls_client=client, **destination)
        scope = tracing_context(enabled=True, client=client, parent=parent)

    try:
        with scope:
            manager = CallbackManager.configure()
            run = manager.on_chain_start({}, {})
            run.on_chain_end({})
        payload = client.create_run.call_args.kwargs
        assert payload.get("address") == destination.get("address")
        assert payload.get("session_name") == destination.get("project_name")
    finally:
        if mode == "configure":
            configure(project_name=None, address=None, enabled=None, client=None)


def test_copy_and_context_address() -> None:
    client = MagicMock(spec=Client)
    with tracing_v2_enabled(address=ADDRESS, client=client) as tracer:
        copied = tracer.copy_with_metadata_defaults(metadata={"foo": "bar"})
        assert copied.address == ADDRESS
        assert copied.project_name is None
        manager = CallbackManager.configure()
        run = manager.on_chain_start({}, {})
        run.on_chain_end({})
    assert client.create_run.call_args.kwargs["address"] == ADDRESS


@pytest.mark.parametrize("address", ["invalid", {"agent_id": "agent"}])
def test_invalid_address(address: Any) -> None:
    with pytest.raises(LangSmithUserError):
        LangChainTracer(address=address, client=MagicMock(spec=Client))


@pytest.mark.parametrize("name", ["LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT"])
def test_incomplete_environment(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "staging")
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()
    client = MagicMock(spec=Client)
    tracer = LangChainTracer(client=client)
    run_id = uuid4()
    with pytest.raises(EnvAddressError):
        tracer.on_chain_start({}, {}, run_id=run_id)
    client.create_run.assert_not_called()
    client.update_run.assert_not_called()


def test_parent_destination_overrides_tracer() -> None:
    client = MagicMock(spec=Client)
    tracer = LangChainTracer(project_name="other-project", client=client)
    parent = RunTree(name="parent", address=ADDRESS, ls_client=client)
    tracer.run_map[str(parent.id)] = parent
    tracer.order_map[parent.id] = (parent.trace_id, parent.dotted_order)
    run_id = uuid4()
    tracer.on_chain_start({}, {}, run_id=run_id, parent_run_id=parent.id)
    tracer.on_chain_end({}, run_id=run_id)
    assert parent.child_runs[0].address == ADDRESS
    assert parent.child_runs[0].session_name is None
    assert client.create_run.call_args.kwargs["address"] == ADDRESS


def test_context_conflicting_arguments() -> None:
    client = MagicMock(spec=Client)
    with (
        tracing_v2_enabled(
            project_name="project", address=ADDRESS, client=client
        ) as tracer,
        pytest.raises(LangSmithUserError, match="not both"),
    ):
        tracer.on_chain_start({}, {}, run_id=uuid4())
    client.create_run.assert_not_called()
    client.update_run.assert_not_called()


def test_trace_callbacks_resolve_environment_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "env-agent")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "staging")
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()
    client = MagicMock(spec=Client)
    with tracing_context(client=client):
        callbacks = _get_trace_callbacks()
        assert isinstance(callbacks, list)
        tracer = callbacks[0]
        assert isinstance(tracer, LangChainTracer)
        run_id = uuid4()
        tracer.on_chain_start({}, {}, run_id=run_id)
        tracer.on_chain_end({}, run_id=run_id)
    assert client.create_run.call_args.kwargs["address"] == ENV_ADDRESS


@pytest.mark.parametrize(
    "destination", [{"address": ADDRESS}, {"project_name": "parent"}]
)
def test_tracer_kwargs_parent_destination(destination: dict[str, str]) -> None:
    client = MagicMock(spec=Client)
    parent = RunTree(name="parent", ls_client=client, **destination)
    with tracing_context(
        parent=parent, project_name="ambient", tags=["tag"], metadata={"key": "value"}
    ):
        kwargs = _get_tracer_kwargs()
        assert kwargs["project_name"] == destination.get("project_name")
        assert kwargs["address"] == destination.get("address")
        assert kwargs["client"] is client
        assert kwargs["tags"] == ["tag"]
        assert kwargs["metadata"] == {"key": "value"}
        explicit = _get_tracer_kwargs("explicit")
        assert explicit["project_name"] == "explicit"
        assert explicit["address"] is None


def test_tracer_kwargs_leave_environment_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_PROJECT", "env-project")
    with tracing_context():
        kwargs = _get_tracer_kwargs()
    assert kwargs["project_name"] is None
    assert kwargs["address"] is None


@pytest.mark.parametrize("project_name", [None, "conflicting"])
def test_trace_callbacks_explicit_address(project_name: str | None) -> None:
    client = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=client, project_name="ambient"):
        callbacks = _get_trace_callbacks(project_name, address=ADDRESS)
        assert isinstance(callbacks, list)
        tracer = callbacks[0]
        assert isinstance(tracer, LangChainTracer)
        if project_name is not None:
            with pytest.raises(LangSmithUserError, match="not both"):
                tracer.on_chain_start({}, {}, run_id=uuid4())
            client.create_run.assert_not_called()
        else:
            run_id = uuid4()
            tracer.on_chain_start({}, {}, run_id=run_id)
            tracer.on_chain_end({}, run_id=run_id)
            assert client.create_run.call_args.kwargs["address"] == ADDRESS


def test_chain_group_explicit_address() -> None:
    client = MagicMock(spec=Client)
    with (
        tracing_context(enabled=True, client=client, project_name="ambient"),
        trace_as_chain_group("group", address=ADDRESS) as manager,
    ):
        run = manager.on_chain_start({}, {})
        run.on_chain_end({})
    assert client.create_run.call_count == 2
    for call in client.create_run.call_args_list:
        assert call.kwargs["address"] == ADDRESS
        assert call.kwargs.get("session_name") is None


async def test_async_chain_group_explicit_address() -> None:
    client = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=client, project_name="ambient"):
        async with atrace_as_chain_group("group", address=ADDRESS) as manager:
            run = await manager.on_chain_start({}, {})
            await run.on_chain_end({})
    assert client.create_run.call_count == 2
    for call in client.create_run.call_args_list:
        assert call.kwargs["address"] == ADDRESS
        assert call.kwargs.get("session_name") is None


@pytest.mark.parametrize("root_address", [False, True])
@pytest.mark.parametrize("replica_mode", ["inherit", "project", "address"])
def test_replica_destinations(*, root_address: bool, replica_mode: str) -> None:
    client = MagicMock(spec=Client)
    replica_client = MagicMock(spec=Client)
    replica: WriteReplica = {"client": replica_client, "primary": False}
    replica_address = address.agent("replica-agent", "staging")
    if replica_mode == "project":
        replica["project_name"] = "replica-project"
    elif replica_mode == "address":
        replica["address"] = replica_address
    tracer = LangChainTracer(
        project_name=None if root_address else "root-project",
        address=ADDRESS if root_address else None,
        client=client,
    )
    parent_id, child_id = uuid4(), uuid4()
    with tracing_context(replicas=[replica]):
        tracer.on_chain_start({}, {}, run_id=parent_id)
        tracer.on_chain_start({}, {}, run_id=child_id, parent_run_id=parent_id)
        tracer.on_chain_end({}, run_id=child_id)
        tracer.on_chain_end({}, run_id=parent_id)
    expected_address = (
        replica_address
        if replica_mode == "address"
        else ADDRESS
        if replica_mode == "inherit" and root_address
        else None
    )
    expected_project = (
        "replica-project"
        if replica_mode == "project"
        else "root-project"
        if replica_mode == "inherit" and not root_address
        else None
    )
    client.create_run.assert_not_called()
    client.update_run.assert_not_called()
    assert replica_client.create_run.call_count == 2
    assert replica_client.update_run.call_count == 2
    creates = [call.kwargs for call in replica_client.create_run.call_args_list]
    updates = [call.kwargs for call in replica_client.update_run.call_args_list]
    for payload in creates + updates:
        assert payload.get("address") == expected_address
        assert payload.get("session_name") == expected_project
    assert {payload["run_id"] for payload in updates} == {
        payload["id"] for payload in creates
    }


def test_replica_preserves_primary_ids() -> None:
    client = MagicMock(spec=Client)
    primary_client = MagicMock(spec=Client)
    secondary_client = MagicMock(spec=Client)
    secondary_address = address.agent("secondary-agent", "staging")
    run_id = uuid4()
    tracer = LangChainTracer(address=ADDRESS, client=client)
    with tracing_context(
        replicas=[
            {"client": primary_client, "primary": True},
            {"client": secondary_client, "address": secondary_address},
        ]
    ):
        tracer.on_chain_start({}, {}, run_id=run_id)
        tracer.on_chain_end({}, run_id=run_id)
    assert primary_client.create_run.call_args.kwargs["id"] == run_id
    assert primary_client.update_run.call_args.kwargs["run_id"] == run_id
    assert primary_client.create_run.call_args.kwargs["address"] == ADDRESS
    secondary_id = secondary_client.create_run.call_args.kwargs["id"]
    assert secondary_id != run_id
    assert secondary_client.update_run.call_args.kwargs["run_id"] == secondary_id
    assert secondary_client.create_run.call_args.kwargs["address"] == secondary_address
    client.create_run.assert_not_called()


def test_replica_conflicting_destination() -> None:
    client = MagicMock(spec=Client)
    tracer = LangChainTracer(address=ADDRESS, client=client)
    with (
        tracing_context(replicas=[{"project_name": "conflict", "address": ADDRESS}]),
        pytest.raises(LangSmithUserError, match="not both"),
    ):
        tracer.on_chain_start({}, {}, run_id=uuid4())
    client.create_run.assert_not_called()
    client.update_run.assert_not_called()


def test_addressed_parent_replica_inherited_outside_context() -> None:
    client = MagicMock(spec=Client)
    replica_client = MagicMock(spec=Client)
    parent = RunTree(
        name="parent",
        address=ADDRESS,
        ls_client=client,
        replicas=[{"client": replica_client, "project_name": "replica-project"}],
    )
    tracer = LangChainTracer(client=client)
    tracer.run_map[str(parent.id)] = parent
    tracer.order_map[parent.id] = (parent.trace_id, parent.dotted_order)
    run_id = uuid4()
    tracer.on_chain_start({}, {}, run_id=run_id, parent_run_id=parent.id)
    tracer.on_chain_end({}, run_id=run_id)
    replica_client.create_run.assert_called_once()
    replica_client.update_run.assert_called_once()
    assert (
        replica_client.create_run.call_args.kwargs["session_name"] == "replica-project"
    )
    assert replica_client.create_run.call_args.kwargs.get("address") is None
    assert parent.child_runs[0].address == ADDRESS
    client.create_run.assert_not_called()


@pytest.mark.parametrize("use_address", [False, True])
def test_chain_group_tracing_disabled(*, use_address: bool) -> None:
    client = MagicMock(spec=Client)
    with (
        tracing_context(enabled=False, client=client),
        trace_as_chain_group(
            "group", address=ADDRESS if use_address else None
        ) as manager,
    ):
        assert not any(
            isinstance(handler, LangChainTracer) for handler in manager.handlers
        )
        run = manager.on_chain_start({}, {})
        run.on_chain_end({})
        manager.on_chain_end({})
    assert manager.ended
    client.create_run.assert_not_called()
    client.update_run.assert_not_called()


@pytest.mark.parametrize("use_address", [False, True])
async def test_async_chain_group_tracing_disabled(*, use_address: bool) -> None:
    client = MagicMock(spec=Client)
    with tracing_context(enabled=False, client=client):
        async with atrace_as_chain_group(
            "group", address=ADDRESS if use_address else None
        ) as manager:
            assert not any(
                isinstance(handler, LangChainTracer) for handler in manager.handlers
            )
            run = await manager.on_chain_start({}, {})
            await run.on_chain_end({})
            await manager.on_chain_end({})
    assert manager.ended
    client.create_run.assert_not_called()
    client.update_run.assert_not_called()
