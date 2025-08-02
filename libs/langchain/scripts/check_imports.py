"""Check Imports Script.

Quickly verify that a list of Python files can be loaded by the Python interpreter
without raising any errors. Ran before running more expensive tests. Useful in
Makefiles.

If loading a file fails, the script prints the problematic filename and the detailed
error traceback.
"""

import random
import string
import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            module_name = "".join(
                random.choice(string.ascii_letters)  # noqa: S311
                for _ in range(20)
            )
            SourceFileLoader(module_name, file).load_module()
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
