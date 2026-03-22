"""Check version consistency between `pyproject.toml` and a version file.

Validates that the version in pyproject.toml matches the hardcoded version
variable in a Python source file. Intended for use as a pre-release check
to prevent version mismatches.

Usage:
    python scripts/check_version.py <package_dir>
    python scripts/check_version.py <package_dir> --pattern VERSION --version-file version.py

Arguments:
    package_dir: Path to the package root (must contain pyproject.toml and
        a Python package directory named after the package).

Options:
    --pattern: Variable name to match (default: `__version__`).
    --version-file: Filename inside the Python package (default: `_version.py`).
    --package-name: Python package directory name (default: auto-detected
        from `langchain_*` directories).
"""

import argparse
import re
import sys
from pathlib import Path


def get_pyproject_version(pyproject_path: Path) -> str | None:
    """Extract version from `pyproject.toml`."""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_source_version(version_path: Path, pattern: str) -> str | None:
    """Extract a version variable from a Python source file."""
    content = version_path.read_text(encoding="utf-8")
    match = re.search(rf'^{pattern}\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    """Validate version consistency."""
    parser = argparse.ArgumentParser(description="Check version consistency.")
    parser.add_argument("package_dir", type=Path, help="Path to the package root.")
    parser.add_argument(
        "--pattern", default="__version__", help="Variable name to match."
    )
    parser.add_argument(
        "--version-file", default="_version.py", help="Version filename."
    )
    parser.add_argument(
        "--package-name", default=None, help="Python package directory name."
    )
    args = parser.parse_args()

    package_dir: Path = args.package_dir.resolve()
    pyproject_path = package_dir / "pyproject.toml"

    if args.package_name:
        pkg_dir = package_dir / args.package_name
        if not pkg_dir.is_dir():
            print(f"Error: {pkg_dir} not found")  # noqa: T201
            return 1
    else:
        # Auto-detect the Python package directory (langchain_*).
        pkg_dirs = sorted(package_dir.glob("langchain_*"))
        pkg_dirs = [
            d for d in pkg_dirs if d.is_dir() and not d.name.endswith(".egg-info")
        ]
        if not pkg_dirs:
            print(  # noqa: T201
                f"Error: no langchain_* package directory found in {package_dir}"
            )
            return 1
        pkg_dir = pkg_dirs[0]
    version_path = pkg_dir / args.version_file

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")  # noqa: T201
        return 1

    if not version_path.exists():
        print(f"Error: {version_path} not found")  # noqa: T201
        return 1

    pyproject_version = get_pyproject_version(pyproject_path)
    source_version = get_source_version(version_path, args.pattern)

    if pyproject_version is None:
        print("Error: Could not find version in pyproject.toml")  # noqa: T201
        return 1

    if source_version is None:
        print(f"Error: Could not find {args.pattern} in {version_path}")  # noqa: T201
        return 1

    if pyproject_version != source_version:
        print("Error: Version mismatch detected!")  # noqa: T201
        print(f"  pyproject.toml: {pyproject_version}")  # noqa: T201
        print(f"  {version_path.relative_to(package_dir)}: {source_version}")  # noqa: T201
        return 1

    label = package_dir.name
    print(f"Version check passed ({label}): {pyproject_version}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
