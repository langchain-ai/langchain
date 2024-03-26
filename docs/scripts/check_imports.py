import importlib
import json
import logging
import os
from pathlib import Path
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

DOCS_DIR = Path(os.path.abspath(__file__)).parents[1] / "docs"
import_pattern = re.compile(
    r"import\s+(\w+)|from\s+([\w\.]+)\s+import\s+((?:\w+(?:,\s*)?)+|\(.*?\))",
    re.DOTALL
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
                import_statements.append((match[0], ''))
            else:  # from ___ import statement
                module, items = match[1], match[2]
                items_list = items.replace(' ', '').split(',')
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


def _check_import_statements(import_statements: List[Tuple[str, str]]) -> None:
    for module, item in import_statements:
        try:
            if item:
                try:
                    # submodule
                    full_module_name = f"{module}.{item}"
                    importlib.import_module(full_module_name)
                except ModuleNotFoundError:
                    # attribute
                    imported_module = importlib.import_module(module)
                    getattr(imported_module, item)
            else:
                importlib.import_module(module)
        except Exception as e:
            logger.error(f"Failed to resolve '{item}' in module '{module}'. Error: {e}")


def check_notebooks(directory: str) -> None:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                import_statements = [
                    (module, item)
                    for module, item in _extract_import_statements(notebook_path)
                    if "langchain" in module
                ]
                _check_import_statements(import_statements)


if __name__ == "__main__":
    check_notebooks(DOCS_DIR)
