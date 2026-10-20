"""Check imports script.

Quickly verify that a list of Python files can be loaded by the Python interpreter
without raising any errors. Ran before running more expensive tests. Useful in
Makefiles.

If loading a file fails, the script prints the problematic filename and the detailed
error traceback.
"""

import sys
import traceback
import uuid
from importlib.util import module_from_spec, spec_from_file_location


def _execute_file(file: str) -> None:
    module_name = f"test_module_{uuid.uuid4().hex[:20]}"
    spec = spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {file}"
        raise ImportError(msg)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)


if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            _execute_file(file)
        except Exception:
            has_failure = True
            print(file)
            traceback.print_exc()
            print()

    sys.exit(1 if has_failure else 0)
