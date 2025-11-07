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
            "langchain-core",
            "langgraph",
            "langgraph-prebuilt",
            "pydantic",
        ],
    )
