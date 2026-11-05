"""Regression tests for release minimum dependency selection."""

from pathlib import Path
from unittest.mock import patch

import pytest
from get_min_versions import get_min_version_from_toml


@pytest.mark.parametrize(
    ("versions_for", "expected"),
    [
        ("release", {"langchain-core": "1.4.7", "langchain-openai": "1.1.0"}),
        ("pull_request", {}),
    ],
)
def test_openai_partner_minimum_versions(
    tmp_path: Path, versions_for: str, expected: dict[str, str]
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\ndependencies = ["langchain-core>=1.4.7,<2.0.0", '
        '"langchain-openai>=1.1.0,<2.0.0"]\n'
    )
    versions = {
        "langchain-core": ["1.6.4", "1.4.7", "1.1.0"],
        "langchain-openai": ["1.6.3", "1.1.0", "1.0.0"],
    }
    with patch("get_min_versions.get_pypi_versions", side_effect=versions.__getitem__):
        assert (
            get_min_version_from_toml(str(manifest), versions_for, "3.11") == expected
        )


@pytest.mark.parametrize(
    ("versions_for", "expected"),
    [
        ("release", {"langchain-core": "1.6.2", "langchain": "1.3.15"}),
        ("pull_request", {}),
    ],
)
def test_optional_dependency_minimum_versions(
    tmp_path: Path, versions_for: str, expected: dict[str, str]
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\ndependencies = ["langchain-core>=1.6.2,<2.0.0"]\n'
        "[project.optional-dependencies]\n"
        'experimental = ["langchain>=1.3.15,<2.0.0"]\n'
        'openai = ["langchain-openai"]\n'
        'core = ["langchain-core>=1.7.0"]\n'
    )
    versions = {
        "langchain-core": ["1.7.4", "1.7.0", "1.6.2", "1.0.0"],
        "langchain": ["1.4.8", "1.3.15", "1.0.0"],
    }
    with patch("get_min_versions.get_pypi_versions", side_effect=versions.__getitem__):
        assert (
            get_min_version_from_toml(str(manifest), versions_for, "3.11") == expected
        )
