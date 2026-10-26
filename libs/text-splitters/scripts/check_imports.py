import sys
import traceback
import uuid
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            module_name = f"test_module_{uuid.uuid4().hex[:20]}"
            loader = SourceFileLoader(module_name, file)
            spec = spec_from_loader(module_name, loader)
            if spec is None:
                msg = f"Could not create import spec for {file}"
                raise ImportError(msg)  # noqa: TRY301
            module = module_from_spec(spec)
            loader.exec_module(module)
        except Exception:  # noqa: BLE001
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
