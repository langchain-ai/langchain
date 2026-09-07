import concurrent.futures
import importlib
import subprocess
import sys
from pathlib import Path


def test_importable_all() -> None:
    for path in Path("../core/langchain_core/").glob("*"):
        module_name = path.stem
        if (
            not module_name.startswith(".")
            and path.suffix != ".typed"
            and module_name != "pydantic_v1"
        ):
            module = importlib.import_module("langchain_core." + module_name)
            all_ = getattr(module, "__all__", [])
            for cls_ in all_:
                getattr(module, cls_)


def try_to_import(module_name: str) -> tuple[int, str]:
    """Try to import a module via subprocess."""
    module = importlib.import_module("langchain_core." + module_name)
    all_ = getattr(module, "__all__", [])
    for cls_ in all_:
        getattr(module, cls_)

    result = subprocess.run(
        [sys.executable, "-c", f"import langchain_core.{module_name}"], check=True
    )
    return result.returncode, module_name


def test_importable_all_via_subprocess() -> None:
    """Test import in isolation.

    !!! note
        ImportErrors due to circular imports can be raised for one sequence of imports
        but not another.
    """
    module_names = []
    for path in Path("../core/langchain_core/").glob("*"):
        module_name = path.stem
        if (
            not module_name.startswith(".")
            and path.suffix != ".typed"
            and module_name != "pydantic_v1"
        ):
            module_names.append(module_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(try_to_import, module_name) for module_name in module_names
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Will raise an exception if the callable raised
            code, module_name = result
            if code != 0:
                msg = f"Failed to import {module_name}."
                raise ValueError(msg)


def test_runnables_does_not_import_langsmith() -> None:
    """Importing `langchain_core.runnables` must not pull in the `langsmith` SDK.

    The tracer modules reach `langsmith` through `tracers.schemas`, so importing
    them at module level loads the SDK for every program, including those that
    never trace. Run in a subprocess so the result cannot be affected by imports
    performed by other tests.

    `Runnable` is bound rather than importing the package alone, because
    `langchain_core.runnables` resolves its exports lazily and would not load
    `runnables.base` otherwise.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from langchain_core.runnables import Runnable; "
                "sys.exit(2 if 'langsmith' in sys.modules else 0)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 2:
        msg = (
            "`langsmith` was imported as a side effect of importing "
            "`langchain_core.runnables`"
        )
        raise AssertionError(msg)
    if result.returncode != 0:
        msg = f"Subprocess failed unexpectedly:\n{result.stderr}"
        raise AssertionError(msg)
