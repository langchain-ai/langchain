import glob
import importlib
import subprocess
from pathlib import Path

import pytest


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

        # Test import in isolation
        # Note: ImportErrors due to circular imports can be raised
        # for one sequence of imports but not another.
        result = subprocess.run(
            ["python", "-c", f"import langchain_core.{module_name}"],
        )
        if result.returncode != 0:
            pytest.fail(f"Failed to import {module_name}.")
