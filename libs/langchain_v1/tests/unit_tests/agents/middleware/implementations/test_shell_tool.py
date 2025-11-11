from __future__ import annotations

import time
import gc
import tempfile
from pathlib import Path

import pytest

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools.base import ToolException

from langchain.agents.middleware.shell_tool import (
    HostExecutionPolicy,
    ShellToolMiddleware,
    _SessionResources,
    RedactionRule,
    _ShellToolInput,
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
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

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
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

        middleware._run_shell_tool(resources, {"command": "export FOO=bar"}, tool_call_id=None)
        restart_message = middleware._run_shell_tool(
            resources, {"restart": True}, tool_call_id=None
        )
        assert "restarted" in restart_message.lower()
        resources = middleware._ensure_resources(state)  # reacquire after restart
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
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]
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
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]
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
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]
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


def test_shell_tool_input_validation() -> None:
    """Test _ShellToolInput validation rules."""
    # Both command and restart not allowed
    with pytest.raises(ValueError, match="only one"):
        _ShellToolInput(command="ls", restart=True)

    # Neither command nor restart provided
    with pytest.raises(ValueError, match="requires either"):
        _ShellToolInput()

    # Valid: command only
    valid_cmd = _ShellToolInput(command="ls")
    assert valid_cmd.command == "ls"
    assert not valid_cmd.restart

    # Valid: restart only
    valid_restart = _ShellToolInput(restart=True)
    assert valid_restart.restart is True
    assert valid_restart.command is None


def test_normalize_shell_command_empty() -> None:
    """Test that empty shell command raises an error."""
    with pytest.raises(ValueError, match="at least one argument"):
        ShellToolMiddleware(shell_command=[])


def test_normalize_env_non_string_keys() -> None:
    """Test that non-string environment keys raise an error."""
    with pytest.raises(TypeError, match="must be strings"):
        ShellToolMiddleware(env={123: "value"})  # type: ignore[dict-item]


def test_normalize_env_coercion(tmp_path: Path) -> None:
    """Test that environment values are coerced to strings."""
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace", env={"NUM": 42, "BOOL": True}
    )
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]
        result = middleware._run_shell_tool(
            resources, {"command": "echo $NUM $BOOL"}, tool_call_id=None
        )
        assert "42" in result
        assert "True" in result
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_shell_tool_missing_command_string(tmp_path: Path) -> None:
    """Test that shell tool raises an error when command is not a string."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

        with pytest.raises(ToolException, match="expects a 'command' string"):
            middleware._run_shell_tool(resources, {"command": None}, tool_call_id=None)

        with pytest.raises(ToolException, match="expects a 'command' string"):
            middleware._run_shell_tool(
                resources,
                {"command": 123},  # type: ignore[dict-item]
                tool_call_id=None,
            )
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_tool_message_formatting_with_id(tmp_path: Path) -> None:
    """Test that tool messages are properly formatted with tool_call_id."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

        result = middleware._run_shell_tool(
            resources, {"command": "echo test"}, tool_call_id="test-id-123"
        )

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "test-id-123"
        assert result.name == "shell"
        assert result.status == "success"
        assert "test" in result.content
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_nonzero_exit_code_returns_error(tmp_path: Path) -> None:
    """Test that non-zero exit codes are marked as errors."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

        result = middleware._run_shell_tool(
            resources,
            {"command": "false"},  # Command that exits with 1 but doesn't kill shell
            tool_call_id="test-id",
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Exit code: 1" in result.content
        assert result.artifact["exit_code"] == 1  # type: ignore[index]
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_truncation_by_bytes(tmp_path: Path) -> None:
    """Test that output is truncated by bytes when max_output_bytes is exceeded."""
    policy = HostExecutionPolicy(max_output_bytes=50, command_timeout=5.0)
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace", execution_policy=policy)
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

        result = middleware._run_shell_tool(
            resources, {"command": "python3 -c 'print(\"x\" * 100)'"}, tool_call_id=None
        )

        assert "truncated at 50 bytes" in result.lower()
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_startup_command_failure(tmp_path: Path) -> None:
    """Test that startup command failure raises an error."""
    policy = HostExecutionPolicy(startup_timeout=1.0)
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace", startup_commands=("exit 1",), execution_policy=policy
    )
    state: AgentState = _empty_state()
    with pytest.raises(RuntimeError, match="Startup command.*failed"):
        middleware.before_agent(state, None)


def test_shutdown_command_failure_logged(tmp_path: Path) -> None:
    """Test that shutdown command failures are logged but don't raise."""
    policy = HostExecutionPolicy(command_timeout=1.0)
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace",
        shutdown_commands=("exit 1",),
        execution_policy=policy,
    )
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
    finally:
        # Should not raise despite shutdown command failing
        middleware.after_agent(state, None)


def test_shutdown_command_timeout_logged(tmp_path: Path) -> None:
    """Test that shutdown command timeouts are logged but don't raise."""
    policy = HostExecutionPolicy(command_timeout=0.1)
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace",
        execution_policy=policy,
        shutdown_commands=("sleep 2",),
    )
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
    finally:
        # Should not raise despite shutdown command timing out
        middleware.after_agent(state, None)


def test_ensure_resources_missing_state(tmp_path: Path) -> None:
    """Test that _ensure_resources raises when resources are missing."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    state: AgentState = _empty_state()

    with pytest.raises(ToolException, match="Shell session resources are unavailable"):
        middleware._ensure_resources(state)  # type: ignore[attr-defined]


def test_empty_output_replaced_with_no_output(tmp_path: Path) -> None:
    """Test that empty command output is replaced with '<no output>'."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

        result = middleware._run_shell_tool(
            resources,
            {"command": "true"},  # Command that produces no output
            tool_call_id=None,
        )

        assert "<no output>" in result
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_stderr_output_labeling(tmp_path: Path) -> None:
    """Test that stderr output is properly labeled."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._ensure_resources(state)  # type: ignore[attr-defined]

        result = middleware._run_shell_tool(
            resources, {"command": "echo error >&2"}, tool_call_id=None
        )

        assert "[stderr] error" in result
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


def test_normalize_commands_string_tuple_list(tmp_path: Path) -> None:
    """Test various command normalization formats."""
    # String
    m1 = ShellToolMiddleware(workspace_root=tmp_path / "w1", startup_commands="echo test")
    assert m1._startup_commands == ("echo test",)  # type: ignore[attr-defined]

    # List
    m2 = ShellToolMiddleware(workspace_root=tmp_path / "w2", startup_commands=["echo test", "pwd"])
    assert m2._startup_commands == ("echo test", "pwd")  # type: ignore[attr-defined]

    # Tuple
    m3 = ShellToolMiddleware(workspace_root=tmp_path / "w3", startup_commands=("echo test",))
    assert m3._startup_commands == ("echo test",)  # type: ignore[attr-defined]

    # None
    m4 = ShellToolMiddleware(workspace_root=tmp_path / "w4")
    assert m4._startup_commands == ()  # type: ignore[attr-defined]


def test_async_methods_delegate_to_sync(tmp_path: Path) -> None:
    """Test that async methods properly delegate to sync methods."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        import asyncio

        # Test abefore_agent
        updates = asyncio.run(middleware.abefore_agent(state, None))
        if updates:
            state.update(updates)

        # Test aafter_agent
        asyncio.run(middleware.aafter_agent(state, None))
    finally:
        pass
