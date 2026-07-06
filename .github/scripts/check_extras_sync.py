"""Check that optional extras stay in sync with required dependencies.

When a package appears in both [project.dependencies] and
[project.optional-dependencies], we ensure their version constraints match.
This prevents silent version drift (e.g. bumping a required dep but
forgetting the corresponding extra).
"""

import sys
import tomllib
from pathlib import Path
from re import compile as re_compile

# Matches the package name at the start of a PEP 508 dependency string.
# Stops at the first non-name character; downstream code is responsible for
# stripping extras (`[...]`) and env markers (`; ...`) from the remainder.
_NAME_RE = re_compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _normalize(name: str) -> str:
    """Normalize a package name for equality comparison.

    Lowercases and maps `-` and `.` to `_`. Looser than PEP 503
    (which uses `-` and collapses runs), but sufficient for matching the
    same package across two PEP 508 strings.

    Returns:
        Lowercased, underscore-normalized package name.
    """
    return name.lower().replace("-", "_").replace(".", "_")


def _parse_dep(dep: str) -> tuple[str, str]:
    """Return `(normalized_name, version_spec)` from a PEP 508 string.

    Strips extras (`pkg[async]`), environment markers (`; python_version ...`),
    URL specifiers (`pkg @ git+...`), and whitespace so the returned
    `version_spec` is directly comparable between a required and optional dep.

    Returns:
        Tuple of normalized package name and bare version specifier.

    Raises:
        ValueError: If the dependency string cannot be parsed.
    """
    match = _NAME_RE.match(dep)
    if not match:
        msg = f"Cannot parse dependency: {dep!r}"
        raise ValueError(msg)
    name = match.group(1)
    rest = dep[match.end() :].strip()

    if rest.startswith("["):
        close = rest.find("]")
        if close == -1:
            msg = f"Unclosed extras bracket in dependency: {dep!r}"
            raise ValueError(msg)
        rest = rest[close + 1 :].strip()

    if ";" in rest:
        rest = rest.split(";", 1)[0].strip()

    # URL specifiers have no comparable version; treat as unconstrained.
    if rest.startswith("@"):
        rest = ""

    rest = " ".join(rest.split())
    return _normalize(name), rest


def main(pyproject_path: Path) -> int:
    """Check extras sync and return `0` on pass, `1` on mismatch or parse error."""
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    required: dict[str, str] = {}
    for dep in data.get("project", {}).get("dependencies", []):
        try:
            name, spec = _parse_dep(dep)
        except ValueError as e:
            print(f"::error file={pyproject_path}::{e}")
            return 1
        required[name] = spec

    optional = data.get("project", {}).get("optional-dependencies", {})
    if not optional:
        return 0

    mismatches: list[str] = []
    for group, deps in optional.items():
        for dep in deps:
            try:
                name, spec = _parse_dep(dep)
            except ValueError as e:
                print(f"::error file={pyproject_path}::{e}")
                return 1
            if name in required and spec != required[name]:
                mismatches.append(
                    f"  [{group}] {name}: extra has '{spec}' "
                    f"but required dep has '{required[name]}'"
                )

    if mismatches:
        print(f"Extra / required dependency version mismatch in {pyproject_path}:")
        print("\n".join(mismatches))
        print(
            "\nUpdate the optional extras in [project.optional-dependencies] "
            "to match [project.dependencies]."
        )
        return 1

    print(f"All extras in {pyproject_path} are in sync with required dependencies.")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pyproject.toml")
    raise SystemExit(main(path))
