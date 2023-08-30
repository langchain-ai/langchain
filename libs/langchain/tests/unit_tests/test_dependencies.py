"""A unit test meant to catch accidental introduction of non-optional dependencies."""
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest
import toml

HERE = Path(__file__).parent

PYPROJECT_TOML = HERE / "../../pyproject.toml"


@pytest.fixture()
def poetry_conf() -> Dict[str, Any]:
    """Load the pyproject.toml file."""
    with open(PYPROJECT_TOML) as f:
        return toml.load(f)["tool"]["poetry"]


def test_required_dependencies(poetry_conf: Mapping[str, Any]) -> None:
    """A test that checks if a new non-optional dependency is being introduced.

    If this test is triggered, it means that a contributor is trying to introduce a new
    required dependency. This should be avoided in most situations.
    """
    # Get the dependencies from the [tool.poetry.dependencies] section
    dependencies = poetry_conf["dependencies"]

    is_required = {
        package_name: isinstance(requirements, str)
        or not requirements.get("optional", False)
        for package_name, requirements in dependencies.items()
    }
    required_dependencies = [
        package_name for package_name, required in is_required.items() if required
    ]

    assert sorted(required_dependencies) == [
        "PyYAML",
        "SQLAlchemy",
        "aiohttp",
        "async-timeout",
        "dataclasses-json",
        "langsmith",
        "numexpr",
        "numpy",
        "pydantic",
        "python",
        "requests",
        "tenacity",
    ]

    unrequired_dependencies = [
        package_name for package_name, required in is_required.items() if not required
    ]
    in_extras = [dep for group in poetry_conf["extras"].values() for dep in group]
    assert set(unrequired_dependencies) == set(in_extras)


def test_test_group_dependencies(poetry_conf: Mapping[str, Any]) -> None:
    """Check if someone is attempting to add additional test dependencies.

    Only dependencies associated with test running infrastructure should be added
    to the test group; e.g., pytest, pytest-cov etc.

    Examples of dependencies that should NOT be included: boto3, azure, postgres, etc.
    """

    test_group_deps = sorted(poetry_conf["group"]["test"]["dependencies"])

    assert test_group_deps == [
        "duckdb-engine",
        "freezegun",
        "lark",
        "pandas",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "pytest-dotenv",
        "pytest-mock",
        "pytest-socket",
        "pytest-watcher",
        "responses",
        "syrupy",
    ]


def test_imports() -> None:
    """Test that you can import all top level things okay."""
    from langchain.agents import OpenAIFunctionsAgent  # noqa: F401
    from langchain.callbacks import OpenAICallbackHandler  # noqa: F401
    from langchain.chains import LLMChain  # noqa: F401
    from langchain.chat_models import ChatOpenAI  # noqa: F401
    from langchain.document_loaders import BSHTMLLoader  # noqa: F401
    from langchain.embeddings import OpenAIEmbeddings  # noqa: F401
    from langchain.llms import OpenAI  # noqa: F401
    from langchain.retrievers import VespaRetriever  # noqa: F401
    from langchain.schema import BasePromptTemplate  # noqa: F401
    from langchain.tools import DuckDuckGoSearchResults  # noqa: F401
    from langchain.utilities import SerpAPIWrapper  # noqa: F401
    from langchain.vectorstores import FAISS  # noqa: F401
