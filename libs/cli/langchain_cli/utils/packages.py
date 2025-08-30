"""Packages utilities."""

from pathlib import Path
from typing import Any, TypedDict

from tomlkit import load


def get_package_root(cwd: Path | None = None) -> Path:
    """Get package root directory."""
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
    """Fields from pyproject.toml that are relevant to LangServe.

    Attributes:
        module: The module to import from, tool.langserve.export_module
        attr: The attribute to import from the module, tool.langserve.export_attr
        package_name: The name of the package, tool.poetry.name

    """

    module: str
    attr: str
    package_name: str


def get_langserve_export(filepath: Path) -> LangServeExport:
    """Get LangServe export information from a pyproject.toml file."""
    with filepath.open() as f:
        data: dict[str, Any] = load(f)
    try:
        module = data["tool"]["langserve"]["export_module"]
        attr = data["tool"]["langserve"]["export_attr"]
        package_name = data["tool"]["poetry"]["name"]
    except KeyError as e:
        msg = "Invalid LangServe PyProject.toml"
        raise KeyError(msg) from e
    return LangServeExport(module=module, attr=attr, package_name=package_name)
