"""Check that all imports in a list of files succeed."""

import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            SourceFileLoader("x", file).load_module()
        except Exception:
            has_failure = True
            traceback.print_exc()

    sys.exit(1 if has_failure else 0)
