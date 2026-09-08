"""Tests for the ShellToolMiddleware and associated shell components."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from langchain.agents.middleware._execution import HostExecutionPolicy
from langchain.agents.middleware.shell_tool import (
    ShellSession,
    ShellToolMiddleware,
    _ShellToolInput,
)


def _get_default_shell() -> tuple[str, ...]:
    """Determine the appropriate shell command based on the platform."""
    if sys.platform == "win32":
        # Common installation paths for Git Bash on Windows
        git_bash_paths = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files\Git\bin\sh.exe",
        ]
        for path in git_bash_paths:
            if os.path.exists(path):
                return (path,)
        # Fallback message or default if git bash isn't found in standard locations
        return ("bash.exe",)
    return ("/bin/bash",)


DEFAULT_SHELL = _get_default_shell()


def test_shell_input_validation() -> None:
    """Test that input validation correctly enforces command vs restart rules."""
    # Valid: only command
    inp1 = _ShellToolInput(command="ls")
    assert inp1.command == "ls"
    assert inp1.restart is None

    # Valid: only restart
    inp2 = _ShellToolInput(restart=True)
    assert inp2.command is None
    assert inp2.restart is True

    # Invalid: neither specified
    with pytest.raises(ValueError, match="requires either 'command' or 'restart'"):
        _ShellToolInput()

    # Invalid: both specified
    with pytest.raises(ValueError, match="Specify only one"):
        _ShellToolInput(command="ls", restart=True)


def test_shell_session_basic_execution() -> None:
    """Test basic command execution and state persistence in ShellSession."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        policy = HostExecutionPolicy(command_timeout=5.0)
        session = ShellSession(workspace, policy, DEFAULT_SHELL, {})

        session.start()
        try:
            # Test simple command execution
            res = session.execute("echo 'hello world'", timeout=5.0)
            assert not res.timed_out
            assert res.exit_code == 0
            assert "hello world" in res.output

            # Test persistence (working directory change)
            session.execute("mkdir test_dir && cd test_dir", timeout=5.0)
            res_pwd = session.execute("pwd", timeout=5.0)
            assert "test_dir" in res_pwd.output
        finally:
            session.stop(timeout=2.0)


def test_shell_session_timeout() -> None:
    """Test that long-running commands trigger timeouts correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        policy = HostExecutionPolicy()
        session = ShellSession(workspace, policy, DEFAULT_SHELL, {})

        session.start()
        try:
            res = session.execute("sleep 2", timeout=0.2)
            assert res.timed_out is True
        finally:
            session.stop(timeout=2.0)


def test_shell_middleware_lifecycle() -> None:
    """Test middleware agent hooks for setup and teardown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        middleware = ShellToolMiddleware(workspace_root=tmpdir, shell_command=DEFAULT_SHELL)

        state: dict = {}
        # Test before_agent setup
        resources_dict = middleware.before_agent(state, None)  # type: ignore
        assert resources_dict is not None
        assert "shell_session_resources" in resources_dict

        state.update(resources_dict)

        # Test after_agent teardown
        middleware.after_agent(state, None)  # type: ignore
