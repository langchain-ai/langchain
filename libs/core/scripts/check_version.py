"""Check version consistency between package metadata and generated artifacts.

This script validates that the version defined in pyproject.toml matches the `VERSION`
variable in `langchain_core/version.py`. It also checks checked-in snapshots that embed
`langchain-core` version metadata. Intended for use as a pre-commit hook to prevent
version mismatches.
"""

import re
import sys
from pathlib import Path

# Matches the `langchain-core` version embedded in serialized model metadata,
# e.g. `{'versions': {'langchain-core': '1.4.7'}}`. Intentionally broad: every such
# occurrence in a snapshot is expected to track the released version.
SNAPSHOT_VERSION_PATTERN = re.compile(r"langchain-core': '([^']+)'")


def get_pyproject_version(pyproject_path: Path) -> str | None:
    """Extract version from `pyproject.toml`."""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_version_py_version(version_path: Path) -> str | None:
    """Extract `VERSION` from `version.py`."""
    content = version_path.read_text(encoding="utf-8")
    match = re.search(r'^VERSION\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_snapshot_version_mismatches(
    snapshots_dir: Path, expected_version: str
) -> list[tuple[Path, int, str]]:
    """Find snapshot `langchain-core` version metadata that is out of date."""
    mismatches = []
    for snapshot_path in sorted(snapshots_dir.rglob("*.ambr")):
        # `errors="replace"` keeps a stray non-UTF-8 file from crashing the hook;
        # the version strings we match are ASCII, so decoding is unaffected.
        content = snapshot_path.read_text(encoding="utf-8", errors="replace")
        for match in SNAPSHOT_VERSION_PATTERN.finditer(content):
            version = match.group(1)
            if version == expected_version:
                continue
            line_number = content.count("\n", 0, match.start()) + 1
            mismatches.append((snapshot_path, line_number, version))
    return mismatches


def main() -> int:
    """Validate version consistency."""
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent

    pyproject_path = package_dir / "pyproject.toml"
    version_path = package_dir / "langchain_core" / "version.py"
    # Scoped to this package's snapshots: only core's own `tests/` tree embeds
    # `langchain-core` version metadata today.
    snapshots_dir = package_dir / "tests"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        return 1

    if not version_path.exists():
        print(f"Error: {version_path} not found")
        return 1

    pyproject_version = get_pyproject_version(pyproject_path)
    version_py_version = get_version_py_version(version_path)

    if pyproject_version is None:
        print("Error: Could not find version in pyproject.toml")
        return 1

    if version_py_version is None:
        print("Error: Could not find VERSION in langchain_core/version.py")
        return 1

    if pyproject_version != version_py_version:
        print("Error: Version mismatch detected!")
        print(f"  pyproject.toml: {pyproject_version}")
        print(f"  langchain_core/version.py: {version_py_version}")
        return 1

    snapshot_mismatches = get_snapshot_version_mismatches(
        snapshots_dir, pyproject_version
    )
    if snapshot_mismatches:
        print("Error: Snapshot version mismatch detected!")
        print(f"  expected langchain-core version: {pyproject_version}")
        for snapshot_path, line_number, snapshot_version in snapshot_mismatches:
            relative_path = snapshot_path.relative_to(package_dir)
            print(f"  {relative_path}:{line_number}: {snapshot_version}")
        return 1

    print(f"Version check passed: {pyproject_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
