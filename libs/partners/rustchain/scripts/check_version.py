"""Check version consistency between `pyproject.toml` and `_version.py`."""

import re
import sys
from pathlib import Path


def get_pyproject_version(pyproject_path: Path) -> str | None:
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_version_py_version(version_path: Path) -> str | None:
    content = version_path.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else None


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    pyproject = get_pyproject_version(root / "pyproject.toml")
    version_py = get_version_py_version(root / "langchain_rustchain" / "_version.py")
    if not pyproject or not version_py or pyproject != version_py:
        print(f"MISMATCH: pyproject={pyproject} _version.py={version_py}")  # noqa: T201
        sys.exit(1)
    print(f"OK: {version_py}")  # noqa: T201