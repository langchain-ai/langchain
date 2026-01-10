from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from langchain.agents.middleware import _execution
from langchain.agents.middleware.shell_tool import (
    CodexSandboxExecutionPolicy,
    DockerExecutionPolicy,
    HostExecutionPolicy,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_resource(
    *,
    with_prlimit: bool,
    has_rlimit_as: bool = True,
) -> Any:
    """Create a fake ``resource`` module for testing."""

    class _BaseResource:
        RLIMIT_CPU = 0
        RLIMIT_DATA = 2

        if has_rlimit_as:
            RLIMIT_AS = 1

        def __init__(self) -> None:
            self.prlimit_calls: list[tuple[int, int, tuple[int, int]]] = []
            self.setrlimit_calls: list[tuple[int, tuple[int, int]]] = []

        def setrlimit(self, resource_name: int, limits: tuple[int, int]) -> None:
            self.setrlimit_calls.append((resource_name, limits))

    if with_prlimit:

        class _Resource(_BaseResource):
            def prlimit(self, pid: int, resource_name: int, limits: tuple[int, int]) -> None:
                self.prlimit_calls.append((pid, resource_name, limits))

    else:

        class _Resource(_BaseResource):
            pass

    return _Resource()


def test_host_policy_validations() -> None:
    with pytest.raises(ValueError, match="max_output_lines must be positive"):
        HostExecutionPolicy(max_output_lines=0)

    with pytest.raises(ValueError, match="cpu_time_seconds must be positive if provided"):
        HostExecutionPolicy(cpu_time_seconds=0)

    with pytest.raises(ValueError, match="memory_bytes must be positive if provided"):
        HostExecutionPolicy(memory_bytes=-1)


def test_host_policy_requires_resource_for_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_execution, "_HAS_RESOURCE", False, raising=False)
    with pytest.raises(RuntimeError):
        HostExecutionPolicy(cpu_time_seconds=1)


def test_host_policy_applies_prlimit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_resource = _make_resource(with_prlimit=True)
    monkeypatch.setattr(_execution, "resource", fake_resource, raising=False)
    monkeypatch.setattr(_execution.sys, "platform", "linux")

    recorded: dict[str, Any] = {}

    class DummyProcess:
        pid = 1234

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        recorded["command"] = list(command)
        recorded["env"] = dict(env)
        recorded["cwd"] = cwd
        recorded["preexec_fn"] = preexec_fn
        recorded["start_new_session"] = start_new_session
        return DummyProcess()

    monkeypatch.setattr(_execution, "_launch_subprocess", fake_launch)

    policy = HostExecutionPolicy(cpu_time_seconds=2, memory_bytes=4096)
    env = {"PATH": os.environ.get("PATH", ""), "VAR": "1"}
    process = policy.spawn(workspace=tmp_path, env=env, command=("/bin/sh",))

    assert process is not None
    assert recorded["preexec_fn"] is None
    assert recorded["start_new_session"] is True
    assert fake_resource.prlimit_calls == [
        (1234, fake_resource.RLIMIT_CPU, (2, 2)),
        (1234, fake_resource.RLIMIT_AS, (4096, 4096)),
    ]
    assert fake_resource.setrlimit_calls == []


def test_host_policy_uses_preexec_on_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_resource = _make_resource(with_prlimit=False)
    monkeypatch.setattr(_execution, "resource", fake_resource, raising=False)
    monkeypatch.setattr(_execution.sys, "platform", "darwin")

    captured: dict[str, Any] = {}

    class DummyProcess:
        pid = 4321

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        captured["preexec_fn"] = preexec_fn
        captured["start_new_session"] = start_new_session
        return DummyProcess()

    monkeypatch.setattr(_execution, "_launch_subprocess", fake_launch)

    policy = HostExecutionPolicy(cpu_time_seconds=5, memory_bytes=8192)
    env = {"PATH": os.environ.get("PATH", "")}
    policy.spawn(workspace=tmp_path, env=env, command=("/bin/sh",))

    preexec_fn = captured["preexec_fn"]
    assert callable(preexec_fn)
    assert captured["start_new_session"] is True

    preexec_fn()
    # macOS fallback should use setrlimit
    assert fake_resource.setrlimit_calls == [
        (fake_resource.RLIMIT_CPU, (5, 5)),
        (fake_resource.RLIMIT_AS, (8192, 8192)),
    ]


def test_host_policy_respects_process_group_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_resource = _make_resource(with_prlimit=True)
    monkeypatch.setattr(_execution, "resource", fake_resource, raising=False)
    monkeypatch.setattr(_execution.sys, "platform", "linux")

    recorded: dict[str, Any] = {}

    class DummyProcess:
        pid = 1111

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        recorded["start_new_session"] = start_new_session
        return DummyProcess()

    monkeypatch.setattr(_execution, "_launch_subprocess", fake_launch)

    policy = HostExecutionPolicy(create_process_group=False)
    env = {"PATH": os.environ.get("PATH", "")}
    policy.spawn(workspace=tmp_path, env=env, command=("/bin/sh",))

    assert recorded["start_new_session"] is False


def test_host_policy_falls_back_to_rlimit_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_resource = _make_resource(with_prlimit=True, has_rlimit_as=False)
    monkeypatch.setattr(_execution, "resource", fake_resource, raising=False)
    monkeypatch.setattr(_execution.sys, "platform", "linux")

    class DummyProcess:
        pid = 2222

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        return DummyProcess()

    monkeypatch.setattr(_execution, "_launch_subprocess", fake_launch)

    policy = HostExecutionPolicy(cpu_time_seconds=7, memory_bytes=2048)
    env = {"PATH": os.environ.get("PATH", "")}
    policy.spawn(workspace=tmp_path, env=env, command=("/bin/sh",))

    assert fake_resource.prlimit_calls == [
        (2222, fake_resource.RLIMIT_CPU, (7, 7)),
        (2222, fake_resource.RLIMIT_DATA, (2048, 2048)),
    ]


@pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="codex CLI not available on PATH",
)
def test_codex_policy_spawns_codex_cli(monkeypatch, tmp_path: Path) -> None:
    recorded: dict[str, list[str]] = {}

    class DummyProcess:
        pass

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        recorded["command"] = list(command)
        assert cwd == tmp_path
        assert env["TEST_VAR"] == "1"
        assert preexec_fn is None
        assert not start_new_session
        return DummyProcess()

    monkeypatch.setattr(
        "langchain.agents.middleware._execution._launch_subprocess",
        fake_launch,
    )
    policy = CodexSandboxExecutionPolicy(
        platform="linux",
        config_overrides={"sandbox_permissions": ["disk-full-read-access"]},
    )

    env = {"TEST_VAR": "1"}
    policy.spawn(workspace=tmp_path, env=env, command=("/bin/bash",))

    expected = [
        shutil.which("codex"),
        "sandbox",
        "linux",
        "-c",
        'sandbox_permissions=["disk-full-read-access"]',
        "--",
        "/bin/bash",
    ]
    assert recorded["command"] == expected


def test_codex_policy_auto_platform_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_execution.sys, "platform", "linux")
    policy = CodexSandboxExecutionPolicy(platform="auto")
    assert policy._determine_platform() == "linux"


def test_codex_policy_auto_platform_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_execution.sys, "platform", "darwin")
    policy = CodexSandboxExecutionPolicy(platform="auto")
    assert policy._determine_platform() == "macos"


def test_codex_policy_resolve_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_execution.shutil, "which", lambda _: None)
    policy = CodexSandboxExecutionPolicy(binary="codex")
    with pytest.raises(RuntimeError):
        policy._resolve_binary()


def test_codex_policy_auto_platform_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_execution.sys, "platform", "win32")
    policy = CodexSandboxExecutionPolicy(platform="auto")
    with pytest.raises(RuntimeError):
        policy._determine_platform()


def test_codex_policy_formats_override_values() -> None:
    policy = CodexSandboxExecutionPolicy()
    assert policy._format_override({"a": 1}) == '{"a": 1}'

    class Custom:
        def __str__(self) -> str:
            return "custom"

    assert policy._format_override(Custom()) == "custom"


def test_codex_policy_sorts_config_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_execution.shutil, "which", lambda _: "/usr/bin/codex")
    policy = CodexSandboxExecutionPolicy(
        config_overrides={"b": 2, "a": 1},
        platform="linux",
    )
    command = policy._build_command(("echo",))
    indices = [i for i, part in enumerate(command) if part == "-c"]
    override_values = [command[i + 1] for i in indices]
    assert override_values == ["a=1", "b=2"]


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker CLI not available on PATH",
)
def test_docker_policy_spawns_docker_run(monkeypatch, tmp_path: Path) -> None:
    recorded: dict[str, list[str]] = {}

    class DummyProcess:
        pass

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        recorded["command"] = list(command)
        assert cwd == tmp_path
        assert "PATH" in env  # host environment should retain system PATH
        assert not start_new_session
        return DummyProcess()

    monkeypatch.setattr(
        "langchain.agents.middleware._execution._launch_subprocess",
        fake_launch,
    )
    policy = DockerExecutionPolicy(
        image="ubuntu:22.04",
        memory_bytes=4096,
        extra_run_args=("--ipc", "host"),
    )

    env = {"PATH": "/bin"}
    policy.spawn(workspace=tmp_path, env=env, command=("/bin/bash",))

    command = recorded["command"]
    assert command[0] == shutil.which("docker")
    assert command[1:4] == ["run", "-i", "--rm"]
    assert "--memory" in command
    assert "4096" in command
    assert "-v" in command
    assert any(str(tmp_path) in part for part in command)
    assert "-w" in command
    w_index = command.index("-w")
    assert command[w_index + 1] == str(tmp_path)
    assert "-e" in command
    assert "PATH=/bin" in command
    assert command[-2:] == ["ubuntu:22.04", "/bin/bash"]


def test_docker_policy_rejects_cpu_limit() -> None:
    with pytest.raises(RuntimeError):
        DockerExecutionPolicy(cpu_time_seconds=1)


def test_docker_policy_validates_memory() -> None:
    with pytest.raises(ValueError, match="memory_bytes must be positive if provided"):
        DockerExecutionPolicy(memory_bytes=0)


def test_docker_policy_skips_mount_for_temp_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_execution.shutil, "which", lambda _: "/usr/bin/docker")

    recorded: dict[str, list[str]] = {}

    class DummyProcess:
        pass

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        recorded["command"] = list(command)
        assert cwd == workspace
        return DummyProcess()

    monkeypatch.setattr(_execution, "_launch_subprocess", fake_launch)

    workspace = tmp_path / f"{_execution.SHELL_TEMP_PREFIX}case"
    workspace.mkdir()
    policy = DockerExecutionPolicy(cpus="1.5")
    env = {"PATH": "/bin"}
    policy.spawn(workspace=workspace, env=env, command=("/bin/sh",))

    command = recorded["command"]
    assert "-v" not in command
    assert "-w" in command
    w_index = command.index("-w")
    assert command[w_index + 1] == "/"
    assert "--cpus" in command
    assert "--network" in command
    assert "none" in command
    assert command[-2:] == [policy.image, "/bin/sh"]


def test_docker_policy_validates_cpus() -> None:
    with pytest.raises(ValueError, match="cpus must be a non-empty string when provided"):
        DockerExecutionPolicy(cpus="  ")


def test_docker_policy_validates_user() -> None:
    with pytest.raises(ValueError, match="user must be a non-empty string when provided"):
        DockerExecutionPolicy(user="  ")


def test_docker_policy_read_only_and_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_execution.shutil, "which", lambda _: "/usr/bin/docker")

    recorded: dict[str, list[str]] = {}

    class DummyProcess:
        pass

    def fake_launch(command, *, env, cwd, preexec_fn, start_new_session):
        recorded["command"] = list(command)
        return DummyProcess()

    monkeypatch.setattr(_execution, "_launch_subprocess", fake_launch)

    workspace = tmp_path
    policy = DockerExecutionPolicy(read_only_rootfs=True, user="1000:1000")
    policy.spawn(workspace=workspace, env={"PATH": "/bin"}, command=("/bin/sh",))

    command = recorded["command"]
    assert "--read-only" in command
    assert "--user" in command
    user_index = command.index("--user")
    assert command[user_index + 1] == "1000:1000"


def test_docker_policy_resolve_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_execution.shutil, "which", lambda _: None)
    policy = DockerExecutionPolicy()
    with pytest.raises(RuntimeError):
        policy._resolve_binary()
