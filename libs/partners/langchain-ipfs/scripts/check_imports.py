"""Check that all public symbols in langchain_ipfs are importable."""

import sys
from importlib import import_module

package = import_module("langchain_ipfs")

public_names = [name for name in dir(package) if not name.startswith("_")]

for name in public_names:
    try:
        getattr(package, name)
    except Exception as exc:
        print(f"FAILED: {name} raised {exc}", file=sys.stderr)
        sys.exit(1)

print("All public symbols are importable.")
