"""Script to check if python modules can be imported."""

import random
import string
import sys
import traceback
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            module_name = "".join(
                random.choice(string.ascii_letters) for _ in range(20)
            )
            loader = SourceFileLoader(module_name, file)
            spec = spec_from_loader(module_name, loader)
            assert spec is not None
            module = module_from_spec(spec)
            sys.modules[module_name] = module
            loader.exec_module(module)
        except Exception:
            has_failure = True
            print(file)
            traceback.print_exc()
            print()

    sys.exit(1 if has_failure else 0)
