import ast
import importlib
import inspect
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def find_classes_in_file(file_path: Path) -> list[str]:
    """Parse a Python file to find all class definitions."""
    with open(file_path, "r", encoding="utf-8") as file:
        node = ast.parse(file.read(), filename=file_path)
    classes = [n.name for n in node.body if isinstance(n, ast.ClassDef)]
    return classes


def get_class_namespace(file_path, package_name):
    for i, part in enumerate(file_path.parts[::-1]):  # reverse order
        if part == package_name:
            return ".".join(list(file_path.parts[-i - 1 : -1]) + [file_path.stem])
    raise ValueError(
        f"Package name '{package_name}' not found in file path '{file_path}'"
    )


@lru_cache()
def get_class_by_class_fqn(class_fqn):
    module_name, class_name = class_fqn.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@lru_cache()
def is_derived(base_class_fqn, class_fqn):
    try:
        base_cls = get_class_by_class_fqn(base_class_fqn)
        cls = get_class_by_class_fqn(class_fqn)
    except (AttributeError, ModuleNotFoundError, TypeError, ImportError) as er:
        logger.warning(f"Failed to load classes {base_class_fqn} and {class_fqn}. {er}")
        return False

    if issubclass(cls, base_cls) and cls is not base_cls:
        return True
    return False
