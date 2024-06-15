from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from tomlkit import dump, inline_table, load
from tomlkit.items import InlineTable


def _get_dep_inline_table(path: Path) -> InlineTable:
    dep = inline_table()
    dep.update({"path": str(path), "develop": True})
    return dep


def add_dependencies_to_pyproject_toml(
    pyproject_toml: Path, local_editable_dependencies: Iterable[Tuple[str, Path]]
) -> None:
    """Add dependencies to pyproject.toml."""
    with open(pyproject_toml, encoding="utf-8") as f:
        # tomlkit types aren't amazing - treat as Dict instead
        pyproject: Dict[str, Any] = load(f)
        pyproject["tool"]["poetry"]["dependencies"].update(
            {
                name: _get_dep_inline_table(loc.relative_to(pyproject_toml.parent))
                for name, loc in local_editable_dependencies
            }
        )
    with open(pyproject_toml, "w", encoding="utf-8") as f:
        dump(pyproject, f)


def remove_dependencies_from_pyproject_toml(
    pyproject_toml: Path, local_editable_dependencies: Iterable[str]
) -> None:
    """Remove dependencies from pyproject.toml."""
    with open(pyproject_toml, encoding="utf-8") as f:
        pyproject: Dict[str, Any] = load(f)
        # tomlkit types aren't amazing - treat as Dict instead
        dependencies = pyproject["tool"]["poetry"]["dependencies"]
        for name in local_editable_dependencies:
            try:
                del dependencies[name]
            except KeyError:
                pass
    with open(pyproject_toml, "w", encoding="utf-8") as f:
        dump(pyproject, f)
