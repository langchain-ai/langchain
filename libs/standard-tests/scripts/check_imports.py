"""Check imports script."""

import secrets
import string
import sys
import traceback
from importlib.util import module_from_spec, spec_from_file_location


def _execute_file(file: str) -> None:
    module_name = "".join(secrets.choice(string.ascii_letters) for _ in range(20))
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
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
