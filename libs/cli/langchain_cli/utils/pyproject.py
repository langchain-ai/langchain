from pathlib import Path
from typing import List

import tomli
import tomli_w


def add_dependencies_to_pyproject_toml(
    pyproject_toml: Path, local_editable_dependencies: List[tuple[str, Path]]
) -> None:
    """Add dependencies to pyproject.toml."""
    with open(pyproject_toml, "rb") as f:
        pyproject = tomli.load(f)
        pyproject["tool"]["poetry"]["dependencies"].update(
            {
                name: {
                    "path": str(loc.relative_to(pyproject_toml.parent)),
                    "develop": True,
                }
                for name, loc in local_editable_dependencies
            }
        )
    with open(pyproject_toml, "wb") as f:
        tomli_w.dump(pyproject, f)


def remove_dependencies_from_pyproject_toml(
    pyproject_toml: Path, local_editable_dependencies: List[str]
) -> None:
    """Remove dependencies from pyproject.toml."""
    with open(pyproject_toml, "rb") as f:
        pyproject = tomli.load(f)
        dependencies = pyproject["tool"]["poetry"]["dependencies"]
        for name in local_editable_dependencies:
            try:
                del dependencies[name]
            except KeyError:
                pass
    with open(pyproject_toml, "wb") as f:
        tomli_w.dump(pyproject, f)
