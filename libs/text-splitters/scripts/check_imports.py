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
        except Exception:  # noqa: BLE001
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
