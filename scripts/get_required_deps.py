#!/bin/python
"""Print the required dependencies."""
from typing import List, Optional, Tuple

import toml

_DEFAULT_DEP_GROUPS: List[Tuple[str, ...]] = [
    ("tool", "poetry", "dependencies"),
    ("tool", "poetry", "group", "test", "dependencies"),
]


def convert_caret_version(version: str) -> str:
    """Convert poetry caret version to pip-friendly format."""
    if "^" in version:
        base_version = version.replace("^", "")
        version_parts = base_version.split(".")
        non_zero_index = next(
            (i for i, part in enumerate(version_parts) if int(part) != 0),
            len(version_parts) - 1,
        )
        max_version = (
            version_parts[:non_zero_index]
            + [str(int(version_parts[non_zero_index]) + 1)]
            + ["0"] * (len(version_parts) - non_zero_index - 1)
        )
        return f'>={base_version},<{".".join(max_version)}'
    return version


def main(
    pyproject_path: Optional[str] = None,
    dependency_groups: Optional[List[Tuple[str, ...]]] = None,
) -> List[str]:
    """Convert dependencies in pyproject.toml to pip-friendly format."""
    pyproject_path = pyproject_path or "pyproject.toml"
    c = toml.load(pyproject_path)
    dependency_groups = dependency_groups or _DEFAULT_DEP_GROUPS
    required_deps = []
    for dep_group in dependency_groups:
        dependencies = c
        for group in dep_group:
            dependencies = dependencies[group]
        required_deps.extend(
            [
                f"{dep}{convert_caret_version(version)}"
                for dep, version in dependencies.items()
                if isinstance(version, str) and dep != "python"
            ]
        )
    return required_deps


if __name__ == "__main__":
    # Print in a pip-friendly format
    required_deps = main()
    print("\n".join(required_deps))
