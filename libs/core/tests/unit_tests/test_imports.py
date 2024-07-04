import concurrent.futures
import glob
import importlib
import subprocess
from pathlib import Path
from typing import Tuple


def test_importable_all() -> None:
    for path in glob.glob("../core/langchain_core/*"):
        relative_path = Path(path).parts[-1]
        if relative_path.endswith(".typed"):
            continue
        module_name = relative_path.split(".")[0]
        module = importlib.import_module("langchain_core." + module_name)
        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            getattr(module, cls_)


def try_to_import(module_name: str) -> Tuple[int, str]:
    """Try to import a module via subprocess."""
    module = importlib.import_module("langchain_core." + module_name)
    all_ = getattr(module, "__all__", [])
    for cls_ in all_:
        getattr(module, cls_)

    result = subprocess.run(
        ["python", "-c", f"import langchain_core.{module_name}"],
    )
    return result.returncode, module_name


def test_importable_all_via_subprocess() -> None:
    """Test import in isolation.

    Note: ImportErrors due to circular imports can be raised
          for one sequence of imports but not another.
    """
    module_names = []
    for path in glob.glob("../core/langchain_core/*"):
        relative_path = Path(path).parts[-1]
        if relative_path.endswith(".typed"):
            continue
        module_name = relative_path.split(".")[0]
        module_names.append(module_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(try_to_import, module_name) for module_name in module_names
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Will raise an exception if the callable raised
            code, module_name = result
            if code != 0:
                raise ValueError(f"Failed to import {module_name}.")
