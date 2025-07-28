"""Import validation script.

Validates that Python files can be imported without errors by dynamically
loading each file as a module. This is crucial for ensuring:

1. All dependencies are properly installed and available
2. Module-level code executes without syntax or runtime errors
3. Import statements are valid and don't create circular dependencies
4. The code structure follows Python import conventions

Typically run in CI/CD to catch import problems before deployment, ensuring that
all modules can be successfully imported in production environments.
"""

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
