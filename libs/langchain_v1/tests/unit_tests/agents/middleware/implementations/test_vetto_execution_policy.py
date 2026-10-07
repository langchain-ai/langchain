from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest

from langchain.agents.middleware._execution import (
    VettoSandboxExecutionPolicy,
)

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path


def test_vetto_policy_validations() -> None:
    with pytest.raises(ValueError, match="Invalid net mode"):
        VettoSandboxExecutionPolicy(net="invalid_mode")

    with pytest.raises(ValueError, match="allowed_domains must be non-empty when net='allowlist'"):
        VettoSandboxExecutionPolicy(net="allowlist", allowed_domains=[])


def test_vetto_policy_spawns_vetto_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: dict[str, Any] = {}

    def fake_launch(
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path,
        preexec_fn: Callable[[], None] | None,
        start_new_session: bool,
    ) -> subprocess.Popen[str]:
        recorded["command"] = list(command)
        assert cwd == tmp_path
        assert env["TEST_VAR"] == "1"
        assert preexec_fn is None
        assert start_new_session is True
        return Mock()

    monkeypatch.setattr(
        "langchain.agents.middleware._execution._launch_subprocess",
        fake_launch,
    )

    policy = VettoSandboxExecutionPolicy(
        net="allowlist",
        allowed_domains=["api.anthropic.com"],
        memory_limit="256MB",
        command_timeout=45.0,
        policy_path="/etc/vetto/policy.toml",
    )
    monkeypatch.setattr(policy, "_resolve_binary", lambda: "/usr/bin/vetto")

    env = {"TEST_VAR": "1"}
    policy.spawn(workspace=tmp_path, env=env, command=("/bin/bash",))

    expected = [
        "/usr/bin/vetto",
        "--ci",
        "--net=allowlist:api.anthropic.com",
        "--timeout=45s",
        "--limits=as=256MB",
        "--policy=/etc/vetto/policy.toml",
        "--",
        "/bin/bash",
    ]
    assert recorded["command"] == expected


def test_vetto_policy_environment_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leaked-key")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    recorded_env: dict[str, str] = {}

    def fake_launch(
        _command: Sequence[str],
        *,
        env: Mapping[str, str],
        **_kwargs: Any,
    ) -> subprocess.Popen[str]:
        nonlocal recorded_env
        recorded_env = dict(env)
        return Mock()

    monkeypatch.setattr(
        "langchain.agents.middleware._execution._launch_subprocess",
        fake_launch,
    )

    policy = VettoSandboxExecutionPolicy()
    monkeypatch.setattr(policy, "_resolve_binary", lambda: "/usr/bin/vetto")

    policy.spawn(workspace=tmp_path, env={"ALLOWED_VAR": "val"}, command=("/bin/sh",))

    assert "OPENAI_API_KEY" not in recorded_env
    assert recorded_env.get("ALLOWED_VAR") == "val"
    assert recorded_env.get("PATH") == "/usr/bin:/bin"


def test_vetto_policy_missing_binary_raises_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = VettoSandboxExecutionPolicy(allow_fallback=False)
    monkeypatch.setattr(policy, "_resolve_binary", lambda: None)
    with pytest.raises(RuntimeError, match="Vetto sandbox policy requires the 'vetto' CLI"):
        policy.spawn(workspace=tmp_path, env={}, command=("/bin/sh",))


def test_vetto_policy_missing_binary_with_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: dict[str, Any] = {}

    def fake_launch(
        command: Sequence[str],
        *_args: Any,
        **_kwargs: Any,
    ) -> subprocess.Popen[str]:
        recorded["command"] = list(command)
        return Mock()

    monkeypatch.setattr(
        "langchain.agents.middleware._execution._launch_subprocess",
        fake_launch,
    )

    policy = VettoSandboxExecutionPolicy(allow_fallback=True)
    monkeypatch.setattr(policy, "_resolve_binary", lambda: None)

    policy.spawn(workspace=tmp_path, env={}, command=("/bin/sh", "-c", "echo ok"))
    assert recorded["command"] == ["/bin/sh", "-c", "echo ok"]


def test_vetto_policy_binary_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = VettoSandboxExecutionPolicy(binary="vetto")

    # 1. VETTO_PATH env var
    monkeypatch.setenv("VETTO_PATH", "/opt/vetto/bin/vetto")
    with patch("pathlib.Path.is_file", return_value=True), patch("os.access", return_value=True):
        assert policy._resolve_binary() == "/opt/vetto/bin/vetto"

    # 2. PATH search
    monkeypatch.delenv("VETTO_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/vetto")
    assert policy._resolve_binary() == "/usr/local/bin/vetto"
