"""Check imports script.

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
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from typing import cast

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            module_name = "".join(
                random.choice(string.ascii_letters)  # noqa: S311
                for _ in range(20)
            )
            loader = SourceFileLoader(module_name, file)
            spec = cast("ModuleSpec", spec_from_loader(module_name, loader))
            module = module_from_spec(spec)
            sys.modules[module_name] = module
            loader.exec_module(module)
        except Exception:
            has_failure = True
            print(file)
            traceback.print_exc()
            print()

    sys.exit(1 if has_failure else 0)
