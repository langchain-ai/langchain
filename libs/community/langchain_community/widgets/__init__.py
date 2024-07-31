"""**Widgets** provide a natural language interface to widgetdatabases."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.widgets.yfiles_widget import (
        YfilesJupyterWidget,
    )


__all__ = [
    "yfiles_jupyter_graphs",
]

_module_lookup = {
    "YfilesJupyterWidget": "langchain_community.widgets.yfiles_widget",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
