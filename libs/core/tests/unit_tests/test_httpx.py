"""Tests for the HTTPX backend selection shim (`langchain_core._httpx`)."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import langchain_core
import langchain_core._httpx
import langchain_core._security._transport

_PACKAGE_DIR = Path(langchain_core.__file__).parent
_SHIM_PATH = _PACKAGE_DIR / "_httpx.py"
_BACKEND_MODULES = {"httpx", "httpx2"}


def test_prefers_httpx2_when_installed() -> None:
    pytest.importorskip("httpx2")
    assert langchain_core._httpx.httpx.__name__ == "httpx2"


def test_falls_back_to_httpx_without_httpx2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx2", None)
    try:
        shim = importlib.reload(langchain_core._httpx)
        assert shim.httpx.__name__ == "httpx"
    finally:
        monkeypatch.undo()
        importlib.reload(langchain_core._httpx)


def test_httpx2_connection_layer_imported_eagerly() -> None:
    """Building the first async transport must not import on the event loop.

    `httpx2` defers importing `httpcore2` until a transport is built, so the shim
    imports it up front. Checked in a fresh interpreter, since other packages in
    this process may already have imported it.
    """
    pytest.importorskip("httpx2")
    code = "import sys; import langchain_core._httpx; assert 'httpcore2' in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_httpx2_selected_without_httpcore2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Platforms such as emscripten ship `httpx2` without `httpcore2`."""
    pytest.importorskip("httpx2")
    monkeypatch.setitem(sys.modules, "httpcore2", None)
    try:
        shim = importlib.reload(langchain_core._httpx)
        assert shim.httpx.__name__ == "httpx2"
    finally:
        monkeypatch.undo()
        importlib.reload(langchain_core._httpx)


def test_only_httpx_is_exported() -> None:
    assert langchain_core._httpx.__all__ == ["httpx"]


def test_transport_uses_shim_backend() -> None:
    transport_globals = vars(langchain_core._security._transport)
    assert transport_globals["httpx"] is langchain_core._httpx.httpx


def _imported_backends(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules & _BACKEND_MODULES


def test_backends_imported_only_through_shim() -> None:
    """Mixing backends would hand `httpx` objects to `httpx2` code or vice versa."""
    offenders = sorted(
        str(path.relative_to(_PACKAGE_DIR))
        for path in _PACKAGE_DIR.rglob("*.py")
        if path != _SHIM_PATH and _imported_backends(path)
    )
    assert offenders == []
