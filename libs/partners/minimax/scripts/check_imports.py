"""This module checks if the given python files can be imported without error."""

import sys
import traceback
from importlib.machinery import SourceFileLoader


def _check_file(file: str) -> bool:
    """Attempt to import a file and return True if it fails."""
    try:
        SourceFileLoader("x", file).load_module()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return True
    return False


if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = any(_check_file(file) for file in files)
    sys.exit(1 if has_failure else 0)
