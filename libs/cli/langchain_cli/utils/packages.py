"""Packages utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, cast

from tomlkit import load


def get_package_root(cwd: Path | None = None) -> Path:
    """Get package root directory.

    Args:
        cwd: The current working directory to start the search from.
            If None, uses the current working directory of the process.

    Returns:
        The path to the package root directory.

    Raises:
        FileNotFoundError: If no `pyproject.toml` file is found in the directory
            hierarchy.
    """
    # traverse path for routes to host (any directory holding a pyproject.toml file)
    package_root = Path.cwd() if cwd is None else cwd
    visited: set[Path] = set()
    while package_root not in visited:
        visited.add(package_root)

        pyproject_path = package_root / "pyproject.toml"
        if pyproject_path.exists():
            return package_root
        package_root = package_root.parent
    msg = "No pyproject.toml found"
    raise FileNotFoundError(msg)


class LangServeExport(TypedDict):
    """Fields from `pyproject.toml` that are relevant to LangServe.

    Attributes:
        module: The module to import from, `tool.langserve.export_module`
        attr: The attribute to import from the module, `tool.langserve.export_attr`
        package_name: The name of the package, `tool.poetry.name`
    """

    module: str
    attr: str
    package_name: str


def get_langserve_export(filepath: Path) -> LangServeExport:
    """Get LangServe export information from a `pyproject.toml` file.

    Args:
        filepath: Path to the `pyproject.toml` file.

    Returns:
        The LangServeExport information.

    Raises:
        KeyError: If the `pyproject.toml` file is missing required fields.
    """
    with filepath.open() as f:
        # tomlkit types aren't amazing - treat as Dict instead
        data = cast("dict[str, Any]", load(f))
    try:
        module = str(data["tool"]["langserve"]["export_module"])
        attr = str(data["tool"]["langserve"]["export_attr"])
        package_name = str(data["tool"]["poetry"]["name"])
    except KeyError as e:
        msg = "Invalid LangServe PyProject.toml"
        raise KeyError(msg) from e
    return LangServeExport(module=module, attr=attr, package_name=package_name)
