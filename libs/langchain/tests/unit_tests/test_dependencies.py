"""A unit test meant to catch accidental introduction of non-optional dependencies."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import toml
from packaging.requirements import Requirement

HERE = Path(__file__).parent

PYPROJECT_TOML = HERE / "../../pyproject.toml"


@pytest.fixture
def uv_conf() -> dict[str, Any]:
    """Load the pyproject.toml file."""
    with PYPROJECT_TOML.open() as f:
        return toml.load(f)


def test_required_dependencies(uv_conf: Mapping[str, Any]) -> None:
    """A test that checks if a new non-optional dependency is being introduced.

    If this test is triggered, it means that a contributor is trying to introduce a new
    required dependency. This should be avoided in most situations.
    """
    # Get the dependencies from the [tool.poetry.dependencies] section
    dependencies = uv_conf["project"]["dependencies"]
    required_dependencies = {Requirement(dep).name for dep in dependencies}

    assert sorted(required_dependencies) == sorted(
        [
            "PyYAML",
            "SQLAlchemy",
            "async-timeout",
            "langchain-core",
            "langchain-text-splitters",
            "langsmith",
            "pydantic",
            "requests",
        ],
    )


def test_test_group_dependencies(uv_conf: Mapping[str, Any]) -> None:
    """Check if someone is attempting to add additional test dependencies.

    Only dependencies associated with test running infrastructure should be added
    to the test group; e.g., pytest, pytest-cov etc.

    Examples of dependencies that should NOT be included: boto3, azure, postgres, etc.
    """

    dependencies = uv_conf["dependency-groups"]["test"]
    test_group_deps = {Requirement(dep).name for dep in dependencies}

    assert sorted(test_group_deps) == sorted(
        [
            "duckdb-engine",
            "freezegun",
            "langchain-core",
            "langchain-tests",
            "langchain-text-splitters",
            "langchain-openai",
            "lark",
            "packaging",
            "pandas",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-dotenv",
            "pytest-mock",
            "pytest-socket",
            "pytest-watcher",
            "pytest-xdist",
            "blockbuster",
            "responses",
            "syrupy",
            "toml",
            "requests-mock",
            # TODO: temporary hack since cffi 1.17.1 doesn't work with py 3.9.
            "cffi",
            "numpy",
        ],
    )
