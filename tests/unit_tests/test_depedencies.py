"""A unit test meant to catch accidental introduction of non-optional dependencies."""
from pathlib import Path

import toml

HERE = Path(__file__).parent

PYPROJECT_TOML = HERE / "../../pyproject.toml"


def test_required_dependencies() -> None:
    """A test that checks if a new non-optional dependency is being introduced.

    If this test is triggered, it means that a contributor is trying to introduce a new
    required dependency. This should be avoided in most situations.
    """
    with open(PYPROJECT_TOML) as f:
        pyproject = toml.load(f)

    # Get the dependencies from the [tool.poetry.dependencies] section
    dependencies = pyproject["tool"]["poetry"]["dependencies"]

    required_dependencies = [
        package_name
        for package_name, requirements in dependencies.items()
        if isinstance(requirements, str) or not requirements.get("optional", False)
    ]

    assert sorted(required_dependencies) == [
        "PyYAML",
        "SQLAlchemy",
        "aiohttp",
        "async-timeout",
        "dataclasses-json",
        "numexpr",
        "numpy",
        "openapi-schema-pydantic",
        "pydantic",
        "python",
        "requests",
        "tenacity",
    ]
