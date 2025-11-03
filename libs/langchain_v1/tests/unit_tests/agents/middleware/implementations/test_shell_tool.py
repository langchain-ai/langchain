from __future__ import annotations

import asyncio
import gc
import tempfile
import time
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import ToolException

from langchain.agents.middleware.shell_tool import (
    HostExecutionPolicy,
    RedactionRule,
    ShellToolMiddleware,
    _SessionResources,
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

        def stop(self, timeout: float) -> None:
            self.stopped = True

    session = DummySession()
    tempdir = tempfile.TemporaryDirectory(dir=tmp_path)
    tempdir_path = Path(tempdir.name)
    resources = _SessionResources(session=session, tempdir=tempdir, policy=policy)  # type: ignore[arg-type]
    finalizer = resources.finalizer

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
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]
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
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

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
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

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
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

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
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

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
    with pytest.raises(RuntimeError, match=r"Startup command.*failed"):
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


def test_empty_output_replaced_with_no_output(tmp_path: Path) -> None:
    """Test that empty command output is replaced with '<no output>'."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()
        updates = middleware.before_agent(state, None)
        if updates:
            state.update(updates)
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

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
        resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

        result = middleware._run_shell_tool(
            resources, {"command": "echo error >&2"}, tool_call_id=None
        )

        assert "[stderr] error" in result
    finally:
        updates = middleware.after_agent(state, None)
        if updates:
            state.update(updates)


@pytest.mark.parametrize(
    ("startup_commands", "expected"),
    [
        ("echo test", ("echo test",)),  # String
        (["echo test", "pwd"], ("echo test", "pwd")),  # List
        (("echo test",), ("echo test",)),  # Tuple
        (None, ()),  # None
    ],
)
def test_normalize_commands_string_tuple_list(
    tmp_path: Path,
    startup_commands: str | list[str] | tuple[str, ...] | None,
    expected: tuple[str, ...],
) -> None:
    """Test various command normalization formats."""
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace", startup_commands=startup_commands
    )
    assert middleware._startup_commands == expected  # type: ignore[attr-defined]


def test_async_methods_delegate_to_sync(tmp_path: Path) -> None:
    """Test that async methods properly delegate to sync methods."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state: AgentState = _empty_state()

        # Test abefore_agent
        updates = asyncio.run(middleware.abefore_agent(state, None))
        if updates:
            state.update(updates)

        # Test aafter_agent
        asyncio.run(middleware.aafter_agent(state, None))
    finally:
        pass


def test_shell_middleware_resumable_after_interrupt(tmp_path: Path) -> None:
    """Test that shell middleware is resumable after an interrupt.

    This test simulates a scenario where:
    1. The middleware creates a shell session
    2. A command is executed
    3. The agent is interrupted (state is preserved)
    4. The agent resumes with the same state
    5. The shell session is reused (not recreated)
    """
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(workspace_root=workspace)

    # Simulate first execution (before interrupt)
    state: AgentState = _empty_state()
    updates = middleware.before_agent(state, None)
    if updates:
        state.update(updates)

    # Get the resources and verify they exist
    resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]
    initial_session = resources.session
    initial_tempdir = resources.tempdir

    # Execute a command to set state
    middleware._run_shell_tool(resources, {"command": "export TEST_VAR=hello"}, tool_call_id=None)

    # Simulate interrupt - state is preserved, but we don't call after_agent
    # In a real scenario, the state would be checkpointed here

    # Simulate resumption - call before_agent again with same state
    # This should reuse existing resources, not create new ones
    updates = middleware.before_agent(state, None)
    if updates:
        state.update(updates)

    # Get resources again - should be the same session
    resumed_resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

    # Verify the session was reused (same object reference)
    assert resumed_resources.session is initial_session
    assert resumed_resources.tempdir is initial_tempdir

    # Verify the session state persisted (environment variable still set)
    result = middleware._run_shell_tool(
        resumed_resources, {"command": "echo ${TEST_VAR:-unset}"}, tool_call_id=None
    )
    assert "hello" in result
    assert "unset" not in result

    # Clean up
    middleware.after_agent(state, None)


def test_get_or_create_resources_creates_when_missing(tmp_path: Path) -> None:
    """Test that _get_or_create_resources creates resources when they don't exist."""
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(workspace_root=workspace)

    state: AgentState = _empty_state()

    # State has no resources initially
    assert "shell_session_resources" not in state

    # Call _get_or_create_resources - should create new resources
    resources = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

    assert isinstance(resources, _SessionResources)
    assert resources.session is not None
    assert state.get("shell_session_resources") is resources

    # Clean up
    resources._finalizer()


def test_get_or_create_resources_reuses_existing(tmp_path: Path) -> None:
    """Test that _get_or_create_resources reuses existing resources."""
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(workspace_root=workspace)

    state: AgentState = _empty_state()

    # Create resources first time
    resources1 = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

    # Call again - should return the same resources
    resources2 = middleware._get_or_create_resources(state)  # type: ignore[attr-defined]

    assert resources1 is resources2
    assert resources1.session is resources2.session

    # Clean up
    resources1._finalizer()
