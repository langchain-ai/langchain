"""Check that the version in __init__.py matches pyproject.toml."""

import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
init_file = root / "langchain_ipfs" / "_version.py"
pyproject = root / "pyproject.toml"

init_version = None
for line in init_file.read_text().splitlines():
    if line.startswith("__version__"):
        init_version = re.search(r'"([^"]+)"', line)
        break

pyproject_version = None
for line in pyproject.read_text().splitlines():
    if line.startswith("version ="):
        pyproject_version = re.search(r'"([^"]+)"', line)
        break

if init_version and pyproject_version:
    if init_version.group(1) != pyproject_version.group(1):
        print(
            f"MISMATCH: __version__ = {init_version.group(1)} "
            f"but pyproject.toml = {pyproject_version.group(1)}",
            file=sys.stderr,
        )
        import sys

        sys.exit(1)
    print(f"Version matches: {init_version.group(1)}")
else:
    print("Could not find version strings", file=sys.stderr)
    import sys

    sys.exit(1)
