"""Unit tests for `scripts/check_version.py`.

The version-consistency check is duplicated verbatim across partner packages, so
exercising the parsing helpers here covers the shared logic. The script is loaded
by path because `scripts/` is not an importable package.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "check_version.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_check_version", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_version = _load_script()


def test_get_pyproject_version_parses_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8")
    assert check_version.get_pyproject_version(pyproject) == "1.2.3"


def test_get_pyproject_version_missing_returns_none(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n', encoding="utf-8")
    assert check_version.get_pyproject_version(pyproject) is None


def test_get_version_py_version_parses_version(tmp_path: Path) -> None:
    version_py = tmp_path / "_version.py"
    version_py.write_text('__version__ = "4.5.6"\n', encoding="utf-8")
    assert check_version.get_version_py_version(version_py) == "4.5.6"


def test_get_version_py_version_missing_returns_none(tmp_path: Path) -> None:
    version_py = tmp_path / "_version.py"
    version_py.write_text('"""No version here."""\n', encoding="utf-8")
    assert check_version.get_version_py_version(version_py) is None


def test_main_passes_for_real_package() -> None:
    """The shipped `pyproject.toml` and `_version.py` must already agree."""
    assert check_version.main() == 0


def test_main_reports_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A version mismatch must fail loudly with a non-zero exit code."""
    monkeypatch.setattr(check_version, "get_pyproject_version", lambda _: "1.0.0")
    monkeypatch.setattr(check_version, "get_version_py_version", lambda _: "2.0.0")
    assert check_version.main() == 1
