"""Check `langchain-typesafe` version consistency."""

import re
import sys
from pathlib import Path


def _read_version(path: Path, pattern: str) -> str | None:
    content = path.read_text(encoding="utf-8")
    match = re.search(pattern, content, re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    """Return a nonzero status when package versions differ."""
    package_dir = Path(__file__).parent.parent
    pyproject_version = _read_version(
        package_dir / "pyproject.toml",
        r'^version\s*=\s*"([^"]+)"',
    )
    module_version = _read_version(
        package_dir / "langchain_typesafe" / "_version.py",
        r'^__version__\s*=\s*"([^"]+)"',
    )
    if pyproject_version != module_version or pyproject_version is None:
        print("Error: package versions do not match.")  # noqa: T201
        return 1
    print(f"Version check passed: {pyproject_version}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
