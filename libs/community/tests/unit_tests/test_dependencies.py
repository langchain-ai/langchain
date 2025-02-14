"""A unit test meant to catch accidental introduction of non-optional dependencies."""

from pathlib import Path
from typing import Any, Dict, Mapping

import pytest
import toml
from packaging.requirements import Requirement

HERE = Path(__file__).parent

PYPROJECT_TOML = HERE / "../../pyproject.toml"


@pytest.fixture()
def uv_conf() -> Dict[str, Any]:
    """Load the pyproject.toml file."""
    with open(PYPROJECT_TOML) as f:
        return toml.load(f)


def test_required_dependencies(uv_conf: Mapping[str, Any]) -> None:
    """A test that checks if a new non-optional dependency is being introduced.

    If this test is triggered, it means that a contributor is trying to introduce a new
    required dependency. This should be avoided in most situations.
    """
    # Get the dependencies from the [tool.poetry.dependencies] section
    dependencies = uv_conf["project"]["dependencies"]
    required_dependencies = set(Requirement(dep).name for dep in dependencies)

    assert sorted(required_dependencies) == sorted(
        [
            "PyYAML",
            "SQLAlchemy",
            "aiohttp",
            "dataclasses-json",
            "httpx-sse",
            "langchain-core",
            "langsmith",
            "numpy",
            "requests",
            "pydantic-settings",
            "tenacity",
            "langchain",
        ]
    )


def test_test_group_dependencies(uv_conf: Mapping[str, Any]) -> None:
    """Check if someone is attempting to add additional test dependencies.

    Only dependencies associated with test running infrastructure should be added
    to the test group; e.g., pytest, pytest-cov etc.

    Examples of dependencies that should NOT be included: boto3, azure, postgres, etc.
    """

    dependencies = uv_conf["dependency-groups"]["test"]
    test_group_deps = set(Requirement(dep).name for dep in dependencies)

    assert sorted(test_group_deps) == sorted(
        [
            "duckdb-engine",
            "freezegun",
            "langchain-core",
            "langchain-tests",
            "langchain",
            "lark",
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
            # TODO: Hack to get around cffi 1.17.1 not working with py3.9, remove when
            # fix is released.
            "cffi",
        ]
    )


def test_imports() -> None:
    """Test that you can import all top level things okay."""
    from langchain_core.prompts import BasePromptTemplate  # noqa: F401

    from langchain_community.callbacks import OpenAICallbackHandler  # noqa: F401
    from langchain_community.chat_models import ChatOpenAI  # noqa: F401
    from langchain_community.document_loaders import BSHTMLLoader  # noqa: F401
    from langchain_community.embeddings import OpenAIEmbeddings  # noqa: F401
    from langchain_community.llms import OpenAI  # noqa: F401
    from langchain_community.retrievers import VespaRetriever  # noqa: F401
    from langchain_community.tools import DuckDuckGoSearchResults  # noqa: F401
    from langchain_community.utilities import (
        SearchApiAPIWrapper,  # noqa: F401
        SerpAPIWrapper,  # noqa: F401
    )
    from langchain_community.vectorstores import FAISS  # noqa: F401
