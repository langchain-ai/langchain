"""Test that package version is consistent across configuration files."""

from pathlib import Path

import toml

import langchain


def test_version_matches_pyproject() -> None:
    """Verify that __version__ in __init__.py matches version in pyproject.toml."""
    # Get the version from the package __init__.py
    init_version = langchain.__version__

    # Read the version from pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with pyproject_path.open() as f:
        pyproject_data = toml.load(f)

    pyproject_version = pyproject_data["project"]["version"]

    # Assert they match
    assert init_version == pyproject_version, (
        f"Version mismatch: __init__.py has '{init_version}' but "
        f"pyproject.toml has '{pyproject_version}'. "
        f"Please update langchain/__init__.py to match pyproject.toml."
    )
