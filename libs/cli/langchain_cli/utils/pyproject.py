from pathlib import Path
from typing import Iterable

from tomlkit import dump, inline_table, load


def _get_dep_inline_table(path: Path):
    dep = inline_table()
    dep.update({"path": str(path), "develop": True})
    return dep


def add_dependencies_to_pyproject_toml(
    pyproject_toml: Path, local_editable_dependencies: Iterable[tuple[str, Path]]
) -> None:
    """Add dependencies to pyproject.toml."""
    with open(pyproject_toml) as f:
        pyproject = load(f)
        pyproject["tool"]["poetry"]["dependencies"].update(
            {
                name: _get_dep_inline_table(loc.relative_to(pyproject_toml.parent))
                for name, loc in local_editable_dependencies
            }
        )
    with open(pyproject_toml, "w") as f:
        dump(pyproject, f)


def remove_dependencies_from_pyproject_toml(
    pyproject_toml: Path, local_editable_dependencies: Iterable[str]
) -> None:
    """Remove dependencies from pyproject.toml."""
    with open(pyproject_toml) as f:
        pyproject = load(f)
        dependencies = pyproject["tool"]["poetry"]["dependencies"]
        for name in local_editable_dependencies:
            try:
                del dependencies[name]
            except KeyError:
                pass
    with open(pyproject_toml, "w") as f:
        dump(pyproject, f)
