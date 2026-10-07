from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from langchain.tools.vetto import VettoProcessInput, VettoProcessTool, VettoShellTool

if TYPE_CHECKING:
    from pathlib import Path


def test_vetto_tool_defaults(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path))
    assert tool.name == "vetto_process"
    assert tool.net == "off"
    assert tool.timeout == 120
    assert tool.memory_limit is None
    assert tool.allow_fallback is False
    assert tool.args_schema == VettoProcessInput

    shell_tool = VettoShellTool(working_dir=str(tmp_path))
    assert shell_tool.name == "vetto_shell"
    assert shell_tool.net == "off"


def test_vetto_tool_build_command_with_binary(tmp_path: Path) -> None:
    tool = VettoProcessTool(
        working_dir=str(tmp_path),
        net="allowlist",
        allowed_domains=["api.anthropic.com", "pypi.org"],
        allow_write=[str(tmp_path / "extra_write")],
        allow_read=["/etc/ssl/certs"],
        timeout=60,
        memory_limit="512MB",
        vetto_binary="/usr/local/bin/vetto",
    )
    with patch.object(tool, "_resolve_vetto_binary", return_value="/usr/local/bin/vetto"):
        cmd = tool._build_command(["ls", "-la"], cwd=str(tmp_path), timeout=60)

        assert cmd[0] == "/usr/local/bin/vetto"
        assert cmd[1] == "run"
        assert "--net=allowlist" in cmd
        assert "--timeout" in cmd
        assert "60" in cmd
        assert "--memory" in cmd
        assert "512MB" in cmd
        assert "--allow-domain" in cmd
        assert "api.anthropic.com" in cmd
        assert "pypi.org" in cmd
        assert "--allow-read" in cmd
        assert "/etc/ssl/certs" in cmd
        assert "--" in cmd
        assert cmd[-2:] == ["ls", "-la"]


def test_vetto_tool_missing_binary_raises_without_fallback(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), allow_fallback=False)
    with (
        patch.object(tool, "_resolve_vetto_binary", return_value=None),
        pytest.raises(RuntimeError, match="Vetto binary not found"),
    ):
        tool._build_command(["echo", "hi"])


def test_vetto_tool_missing_binary_passes_with_fallback(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), allow_fallback=True)
    with patch.object(tool, "_resolve_vetto_binary", return_value=None):
        cmd = tool._build_command(["echo", "hi"])
        assert cmd == ["echo", "hi"]


def test_vetto_tool_workspace_traversal_prevention(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), allow_fallback=True)
    with pytest.raises(PermissionError, match="escapes configured workspace"):
        tool._run("cat secret.txt", cwd="../../etc")


def test_vetto_tool_successful_run(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), allow_fallback=True)
    fake_result = {
        "exit_code": 0,
        "stdout": "total 0\n",
        "stderr": "",
        "timed_out": False,
        "elapsed_seconds": 0.05,
    }
    with patch.object(tool, "_execute_subprocess", return_value=fake_result):
        output = tool._run("ls")
        assert output == "total 0\n"


def test_vetto_tool_error_exit_code(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), allow_fallback=True)
    fake_result = {
        "exit_code": 1,
        "stdout": "",
        "stderr": "File not found",
        "timed_out": False,
        "elapsed_seconds": 0.02,
    }
    with patch.object(tool, "_execute_subprocess", return_value=fake_result):
        output = tool._run("cat non_existent.txt")
        assert "File not found" in output
        assert "Exit code: 1" in output


def test_vetto_tool_timeout(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), timeout=10, allow_fallback=True)
    fake_result = {
        "exit_code": 124,
        "stdout": "",
        "stderr": "Command timed out",
        "timed_out": True,
        "elapsed_seconds": 10.0,
    }
    with patch.object(tool, "_execute_subprocess", return_value=fake_result):
        output = tool._run("sleep 100")
        assert "Error: Command timed out after 10 seconds." in output


@pytest.mark.asyncio
async def test_vetto_tool_async_run(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), allow_fallback=True)
    fake_result = {
        "exit_code": 0,
        "stdout": "async result\n",
        "stderr": "",
        "timed_out": False,
        "elapsed_seconds": 0.01,
    }
    with patch.object(tool, "_execute_subprocess", return_value=fake_result):
        output = await tool._arun("echo async")
        assert output == "async result\n"


def test_vetto_tool_binary_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path))

    # 1. Explicit binary attribute
    tool.vetto_binary = "/custom/bin/vetto"
    with patch("pathlib.Path.is_file", return_value=True), patch("os.access", return_value=True):
        assert tool._resolve_vetto_binary() == "/custom/bin/vetto"

    # 2. VETTO_PATH environment variable
    tool.vetto_binary = None
    monkeypatch.setenv("VETTO_PATH", "/env/bin/vetto")
    with patch("pathlib.Path.is_file", return_value=True), patch("os.access", return_value=True):
        assert tool._resolve_vetto_binary() == "/env/bin/vetto"

    # 3. PATH search via shutil.which
    monkeypatch.delenv("VETTO_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/vetto")
    assert tool._resolve_vetto_binary() == "/usr/bin/vetto"


def test_vetto_tool_execute_subprocess_timeout(tmp_path: Path) -> None:
    tool = VettoProcessTool(working_dir=str(tmp_path), allow_fallback=True)
    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["sleep"], timeout=1),
        (b"", b"Command timed out"),
    ]

    with patch("subprocess.Popen", return_value=mock_proc), patch("os.killpg") as mock_killpg:
        result = tool._execute_subprocess(["sleep", "10"], timeout=1)
        assert result["timed_out"] is True
        assert result["exit_code"] == 124
        assert mock_killpg.called
