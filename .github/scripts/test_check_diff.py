"""Regression tests for supported CI interpreters."""

from pathlib import Path

import pytest
from check_diff import _get_configs_for_single_dir


@pytest.mark.parametrize(
    ("job", "directory", "minimum"),
    [
        ("test", "libs/partners/openai", "3.10"),
        ("compile-integration-tests", "libs/partners/openai", "3.11"),
        ("test", "libs/partners/chroma", "3.11"),
        ("test", "libs/langchain", "3.10"),
        ("extended-tests", "libs/langchain", "3.10"),
    ],
)
def test_ci_python_versions(
    monkeypatch: pytest.MonkeyPatch, job: str, directory: str, minimum: str
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    configs = _get_configs_for_single_dir(job, directory)
    assert [config["python-version"] for config in configs] == [minimum, "3.14"]
