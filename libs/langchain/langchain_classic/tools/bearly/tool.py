from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import BearlyInterpreterTool
    from langchain_community.tools.bearly.tool import (
        BearlyInterpreterToolArguments,
        FileInfo,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "BearlyInterpreterToolArguments": "langchain_community.tools.bearly.tool",
    "FileInfo": "langchain_community.tools.bearly.tool",
    "BearlyInterpreterTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BearlyInterpreterTool",
    "BearlyInterpreterToolArguments",
    "FileInfo",
]
