from pathlib import Path
from typing import Optional, Set


def get_package_root(cwd: Optional[Path] = None) -> Path:
    # traverse path for routes to host (any directory holding a pyproject.toml file)
    package_root = Path.cwd() if cwd is None else cwd
    visited: Set[Path] = set()
    while package_root not in visited:
        visited.add(package_root)

        pyproject_path = package_root / "pyproject.toml"
        if pyproject_path.exists():
            return package_root
        package_root = package_root.parent
    raise FileNotFoundError("No pyproject.toml found")
