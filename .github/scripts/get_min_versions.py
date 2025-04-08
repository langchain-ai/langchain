from collections import defaultdict
import sys
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    # for python 3.10 and below, which doesnt have stdlib tomllib
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


import requests
from packaging.version import parse
from typing import List

import re


MIN_VERSION_LIBS = [
    "langchain-core",
    "langchain-community",
    "langchain",
    "langchain-text-splitters",
    "numpy",
    "SQLAlchemy",
]

# some libs only get checked on release because of simultaneous changes in
# multiple libs
SKIP_IF_PULL_REQUEST = [
    "langchain-core",
    "langchain-text-splitters",
    "langchain",
    "langchain-community",
]


def get_pypi_versions(package_name: str) -> List[str]:
    """
    Fetch all available versions for a package from PyPI.

    Args:
        package_name (str): Name of the package

    Returns:
        List[str]: List of all available versions

    Raises:
        requests.exceptions.RequestException: If PyPI API request fails
        KeyError: If package not found or response format unexpected
    """
    pypi_url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(pypi_url)
    response.raise_for_status()
    return list(response.json()["releases"].keys())


def get_minimum_version(package_name: str, spec_string: str) -> Optional[str]:
    """
    Find the minimum published version that satisfies the given constraints.

    Args:
        package_name (str): Name of the package
        spec_string (str): Version specification string (e.g., ">=0.2.43,<0.4.0,!=0.3.0")

    Returns:
        Optional[str]: Minimum compatible version or None if no compatible version found
    """
    # rewrite occurrences of ^0.0.z to 0.0.z (can be anywhere in constraint string)
    spec_string = re.sub(r"\^0\.0\.(\d+)", r"0.0.\1", spec_string)
    # rewrite occurrences of ^0.y.z to >=0.y.z,<0.y+1 (can be anywhere in constraint string)
    for y in range(1, 10):
        spec_string = re.sub(rf"\^0\.{y}\.(\d+)", rf">=0.{y}.\1,<0.{y+1}", spec_string)
    # rewrite occurrences of ^x.y.z to >=x.y.z,<x+1.0.0 (can be anywhere in constraint string)
    for x in range(1, 10):
        spec_string = re.sub(
            rf"\^{x}\.(\d+)\.(\d+)", rf">={x}.\1.\2,<{x+1}", spec_string
        )

    spec_set = SpecifierSet(spec_string)
    all_versions = get_pypi_versions(package_name)

    valid_versions = []
    for version_str in all_versions:
        try:
            version = parse(version_str)
            if spec_set.contains(version):
                valid_versions.append(version)
        except ValueError:
            continue

    return str(min(valid_versions)) if valid_versions else None


def _check_python_version_from_requirement(
    requirement: Requirement, python_version: str
) -> bool:
    if not requirement.marker:
        return True
    else:
        marker_str = str(requirement.marker)
        if "python_version" or "python_full_version" in marker_str:
            python_version_str = "".join(
                char
                for char in marker_str
                if char.isdigit() or char in (".", "<", ">", "=", ",")
            )
            return check_python_version(python_version, python_version_str)
        return True


def get_min_version_from_toml(
    toml_path: str,
    versions_for: str,
    python_version: str,
    *,
    include: Optional[list] = None,
):
    # Parse the TOML file
    with open(toml_path, "rb") as file:
        toml_data = tomllib.load(file)

    dependencies = defaultdict(list)
    for dep in toml_data["project"]["dependencies"]:
        requirement = Requirement(dep)
        dependencies[requirement.name].append(requirement)

    # Initialize a dictionary to store the minimum versions
    min_versions = {}

    # Iterate over the libs in MIN_VERSION_LIBS
    for lib in set(MIN_VERSION_LIBS + (include or [])):
        if versions_for == "pull_request" and lib in SKIP_IF_PULL_REQUEST:
            # some libs only get checked on release because of simultaneous
            # changes in multiple libs
            continue
        # Check if the lib is present in the dependencies
        if lib in dependencies:
            if include and lib not in include:
                continue
            requirements = dependencies[lib]
            for requirement in requirements:
                if _check_python_version_from_requirement(requirement, python_version):
                    version_string = str(requirement.specifier)
                    break

            # Use parse_version to get the minimum supported version from version_string
            min_version = get_minimum_version(lib, version_string)

            # Store the minimum version in the min_versions dictionary
            min_versions[lib] = min_version

    return min_versions


def check_python_version(version_string, constraint_string):
    """
    Check if the given Python version matches the given constraints.

    :param version_string: A string representing the Python version (e.g. "3.8.5").
    :param constraint_string: A string representing the package's Python version constraints (e.g. ">=3.6, <4.0").
    :return: True if the version matches the constraints, False otherwise.
    """

    # rewrite occurrences of ^0.0.z to 0.0.z (can be anywhere in constraint string)
    constraint_string = re.sub(r"\^0\.0\.(\d+)", r"0.0.\1", constraint_string)
    # rewrite occurrences of ^0.y.z to >=0.y.z,<0.y+1.0 (can be anywhere in constraint string)
    for y in range(1, 10):
        constraint_string = re.sub(
            rf"\^0\.{y}\.(\d+)", rf">=0.{y}.\1,<0.{y+1}.0", constraint_string
        )
    # rewrite occurrences of ^x.y.z to >=x.y.z,<x+1.0.0 (can be anywhere in constraint string)
    for x in range(1, 10):
        constraint_string = re.sub(
            rf"\^{x}\.0\.(\d+)", rf">={x}.0.\1,<{x+1}.0.0", constraint_string
        )

    try:
        version = Version(version_string)
        constraints = SpecifierSet(constraint_string)
        return version in constraints
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Get the TOML file path from the command line argument
    toml_file = sys.argv[1]
    versions_for = sys.argv[2]
    python_version = sys.argv[3]
    assert versions_for in ["release", "pull_request"]

    # Call the function to get the minimum versions
    min_versions = get_min_version_from_toml(toml_file, versions_for, python_version)

    print(" ".join([f"{lib}=={version}" for lib, version in min_versions.items()]))
