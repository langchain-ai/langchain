from __future__ import annotations

import time
import gc
import tempfile
from pathlib import Path

import pytest

from langchain.agents.middleware.shell_tool import (
    HostExecutionPolicy,
    ShellToolMiddleware,
    _SessionResources,
    RedactionRule,
)
from langchain.agents.middleware.types import AgentState


def _empty_state() -> AgentState:
    return {"messages": []}  # type: ignore[return-value]


def test_executes_command_and_persists_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(workspace_root=workspace)
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

        middleware._run_shell_tool(resources, {"command": "cd /"}, tool_call_id=None)
        result = middleware._run_shell_tool(resources, {"command": "pwd"}, tool_call_id=None)
        assert isinstance(result, str)
        assert result.strip() == "/"
        echo_result = middleware._run_shell_tool(
            resources, {"command": "echo ready"}, tool_call_id=None
        )
        assert "ready" in echo_result
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_restart_resets_session_environment(tmp_path: Path) -> None:
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

        middleware._run_shell_tool(resources, {"command": "export FOO=bar"}, tool_call_id=None)
        restart_message = middleware._run_shell_tool(
            resources, {"restart": True}, tool_call_id=None
        )
        assert "restarted" in restart_message.lower()
        resources = middleware._get_or_create_resources(state)  # reacquire after restart
        result = middleware._run_shell_tool(
            resources, {"command": "echo ${FOO:-unset}"}, tool_call_id=None
        )
        assert "unset" in result
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_truncation_indicator_present(tmp_path: Path) -> None:
    policy = HostExecutionPolicy(max_output_lines=5, command_timeout=5.0)
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace", execution_policy=policy)
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]
        result = middleware._run_shell_tool(resources, {"command": "seq 1 20"}, tool_call_id=None)
        assert "Output truncated" in result
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_timeout_returns_error(tmp_path: Path) -> None:
    policy = HostExecutionPolicy(command_timeout=0.5)
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace", execution_policy=policy)
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]
        start = time.monotonic()
        result = middleware._run_shell_tool(resources, {"command": "sleep 2"}, tool_call_id=None)
        elapsed = time.monotonic() - start
        assert elapsed < policy.command_timeout + 2.0
        assert "timed out" in result.lower()
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_redaction_policy_applies(tmp_path: Path) -> None:
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace",
        redaction_rules=(RedactionRule(pii_type="email", strategy="redact"),),
    )
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]
        message = middleware._run_shell_tool(
            resources,
            {"command": "printf 'Contact: user@example.com\\n'"},
            tool_call_id=None,
        )
        assert "[REDACTED_EMAIL]" in message
        assert "user@example.com" not in message
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_startup_and_shutdown_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(
        workspace_root=workspace,
        startup_commands=("touch startup.txt",),
        shutdown_commands=("touch shutdown.txt",),
    )
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        assert (workspace / "startup.txt").exists()
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)
    assert (workspace / "shutdown.txt").exists()


def test_session_resources_finalizer_cleans_up(tmp_path: Path) -> None:
    policy = HostExecutionPolicy(termination_timeout=0.1)

    class DummySession:
        def __init__(self) -> None:
            self.stopped: bool = False

        def stop(self, timeout: float) -> None:  # noqa: ARG002
            self.stopped = True

    session = DummySession()
    tempdir = tempfile.TemporaryDirectory(dir=tmp_path)
    tempdir_path = Path(tempdir.name)
    resources = _SessionResources(session=session, tempdir=tempdir, policy=policy)  # type: ignore[arg-type]
    finalizer = resources._finalizer

    # Drop our last strong reference and force collection.
    del resources
    gc.collect()

    assert not finalizer.alive
    assert session.stopped
    assert not tempdir_path.exists()
