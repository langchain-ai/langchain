import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    # for python 3.10 and below, which doesnt have stdlib tomllib
    import tomli as tomllib

from packaging.version import parse as parse_version
import re

MIN_VERSION_LIBS = [
    "langchain-core",
    "langchain-community",
    "langchain",
    "langchain-text-splitters",
    "SQLAlchemy",
]

SKIP_IF_PULL_REQUEST = ["langchain-core"]


def get_min_version(version: str) -> str:
    # base regex for x.x.x with cases for rc/post/etc
    # valid strings: https://peps.python.org/pep-0440/#public-version-identifiers
    vstring = r"\d+(?:\.\d+){0,2}(?:(?:a|b|rc|\.post|\.dev)\d+)?"
    # case ^x.x.x
    _match = re.match(f"^\\^({vstring})$", version)
    if _match:
        return _match.group(1)

    # case >=x.x.x,<y.y.y
    _match = re.match(f"^>=({vstring}),<({vstring})$", version)
    if _match:
        _min = _match.group(1)
        _max = _match.group(2)
        assert parse_version(_min) < parse_version(_max)
        return _min

    # case x.x.x
    _match = re.match(f"^({vstring})$", version)
    if _match:
        return _match.group(1)

    raise ValueError(f"Unrecognized version format: {version}")


def get_min_version_from_toml(toml_path: str, versions_for: str):
    # Parse the TOML file
    with open(toml_path, "rb") as file:
        toml_data = tomllib.load(file)

    # Get the dependencies from tool.poetry.dependencies
    dependencies = toml_data["tool"]["poetry"]["dependencies"]

    # Initialize a dictionary to store the minimum versions
    min_versions = {}

    # Iterate over the libs in MIN_VERSION_LIBS
    for lib in MIN_VERSION_LIBS:
        if versions_for == "pull_request" and lib in SKIP_IF_PULL_REQUEST:
            # some libs only get checked on release because of simultaneous
            # changes
            continue
        # Check if the lib is present in the dependencies
        if lib in dependencies:
            # Get the version string
            version_string = dependencies[lib]

            if isinstance(version_string, dict):
                version_string = version_string["version"]

            # Use parse_version to get the minimum supported version from version_string
            min_version = get_min_version(version_string)

            # Store the minimum version in the min_versions dictionary
            min_versions[lib] = min_version

    return min_versions


if __name__ == "__main__":
    # Get the TOML file path from the command line argument
    toml_file = sys.argv[1]
    versions_for = sys.argv[2]
    assert versions_for in ["release", "pull_request"]

    # Call the function to get the minimum versions
    min_versions = get_min_version_from_toml(toml_file, versions_for)

    print(" ".join([f"{lib}=={version}" for lib, version in min_versions.items()]))
