"""Check documentation for broken import statements.

Validates that all import statements in Jupyter notebooks within the documentation
directory are functional and can be successfully imported.

- Scans all `.ipynb` files in `docs/`
- Extracts import statements from code cells
- Tests each import to ensure it works
- Reports any broken imports that would fail for users

Usage:
    python docs/scripts/check_imports.py

Exit codes:
    0: All imports are valid
    1: Found broken imports (ImportError raised)
"""

import importlib
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

DOCS_DIR = Path(os.path.abspath(__file__)).parents[1] / "docs"
import_pattern = re.compile(
    r"import\s+(\w+)|from\s+([\w\.]+)\s+import\s+((?:\w+(?:,\s*)?)+|\(.*?\))", re.DOTALL
)


def _get_imports_from_code_cell(code_lines: str) -> List[Tuple[str, str]]:
    """Get (module, import) statements from a single code cell."""
    import_statements = []
    for line in code_lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        # Join lines that end with a backslash
        if line.endswith("\\"):
            line = line[:-1].rstrip() + " "
            continue
        matches = import_pattern.findall(line)
        for match in matches:
            if match[0]:  # simple import statement
                import_statements.append((match[0], ""))
            else:  # from ___ import statement
                module, items = match[1], match[2]
                items_list = items.replace(" ", "").split(",")
                for item in items_list:
                    import_statements.append((module, item))
    return import_statements


def _extract_import_statements(notebook_path: str) -> List[Tuple[str, str]]:
    """Get (module, import) statements from a Jupyter notebook."""
    with open(notebook_path, "r", encoding="utf-8") as file:
        notebook = json.load(file)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    import_statements = []
    for cell in code_cells:
        code_lines = cell["source"]
        import_statements.extend(_get_imports_from_code_cell(code_lines))
    return import_statements


def _get_bad_imports(import_statements: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Collect offending import statements."""
    offending_imports = []
    for module, item in import_statements:
        try:
            if item:
                try:
                    # submodule
                    full_module_name = f"{module}.{item}"
                    importlib.import_module(full_module_name)
                except ModuleNotFoundError:
                    # attribute
                    try:
                        imported_module = importlib.import_module(module)
                        getattr(imported_module, item)
                    except AttributeError:
                        offending_imports.append((module, item))
                except Exception:
                    offending_imports.append((module, item))
            else:
                importlib.import_module(module)
        except Exception:
            offending_imports.append((module, item))

    return offending_imports


def _is_relevant_import(module: str) -> bool:
    """Check if module is recognized."""
    # Ignore things like langchain_{bla}, where bla is unrecognized.
    recognized_packages = [
        "langchain",
        "langchain_core",
        "langchain_community",
        # "langchain_experimental",
        "langchain_text_splitters",
    ]
    return module.split(".")[0] in recognized_packages


def _serialize_bad_imports(bad_files: list) -> str:
    """Serialize bad imports to a string."""
    bad_imports_str = ""
    for file, bad_imports in bad_files:
        bad_imports_str += f"File: {file}\n"
        for module, item in bad_imports:
            bad_imports_str += f"    {module}.{item}\n"
    return bad_imports_str


def check_notebooks(directory: str) -> list:
    """Check notebooks for broken import statements."""
    bad_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb") and not file.endswith("-checkpoint.ipynb"):
                notebook_path = os.path.join(root, file)
                import_statements = [
                    (module, item)
                    for module, item in _extract_import_statements(notebook_path)
                    if _is_relevant_import(module)
                ]
                bad_imports = _get_bad_imports(import_statements)
                if bad_imports:
                    bad_files.append(
                        (
                            os.path.join(root, file),
                            bad_imports,
                        )
                    )
    return bad_files


if __name__ == "__main__":
    bad_files = check_notebooks(DOCS_DIR)
    if bad_files:
        raise ImportError(f"Found bad imports:\n{_serialize_bad_imports(bad_files)}")
