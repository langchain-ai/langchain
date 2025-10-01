"""Check imports."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        if file.endswith("__init__.py"):
            continue
        try:
            # Import the module
            module_name = file.replace("/", ".").replace(".py", "")
            if module_name.startswith("langchain_baseten"):
                __import__(module_name)
        except Exception as e:
            print(f"Failed to import {file}: {e}")
            has_failure = True
    if has_failure:
        sys.exit(1)
