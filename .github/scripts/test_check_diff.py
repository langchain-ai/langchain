"""Regression tests for CI Python version selection."""

from pathlib import Path

import pytest
from check_diff import _get_configs_for_single_dir


@pytest.mark.parametrize(
    ("directory", "minimum", "expected"),
    [
        ("libs/partners/example", "3.10", ["3.10", "3.14"]),
        ("libs/partners/example", "3.11", ["3.11", "3.14"]),
        ("libs/partners/example", "3.14", ["3.14"]),
        ("libs/core", "3.10", ["3.10", "3.11", "3.12", "3.13", "3.14"]),
    ],
)
def test_package_python_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: str,
    minimum: str,
    expected: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    package = tmp_path / directory
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        f'[project]\nrequires-python = ">={minimum},<4.0.0"\n'
    )
    configs = _get_configs_for_single_dir("test", directory)
    assert [config["python-version"] for config in configs] == expected


def test_extended_tests_respect_editable_dependency_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    package = tmp_path / "libs/langchain"
    partner = tmp_path / "libs/partners/openai"
    package.mkdir(parents=True)
    partner.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.10,<4.0.0"\n'
    )
    (partner / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.11,<4.0.0"\n'
    )
    (package / "extended_testing_deps.txt").write_text(
        "-e ../partners/openai\njsonschema>=4.22.0,<5\n"
    )
    extended = _get_configs_for_single_dir("extended-tests", "libs/langchain")
    regular = _get_configs_for_single_dir("test", "libs/langchain")
    assert [config["python-version"] for config in extended] == ["3.11", "3.14"]
    assert [config["python-version"] for config in regular] == ["3.10", "3.14"]


def test_no_compatible_python_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.15"\n')
    with pytest.raises(ValueError, match="No supported CI Python versions"):
        _get_configs_for_single_dir("test", ".")
