from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utils.ernie_functions import (
        FunctionDescription,
        ToolDescription,
        convert_pydantic_to_ernie_function,
        convert_pydantic_to_ernie_tool,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "FunctionDescription": "langchain_community.utils.ernie_functions",
    "ToolDescription": "langchain_community.utils.ernie_functions",
    "convert_pydantic_to_ernie_function": "langchain_community.utils.ernie_functions",
    "convert_pydantic_to_ernie_tool": "langchain_community.utils.ernie_functions",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "FunctionDescription",
    "ToolDescription",
    "convert_pydantic_to_ernie_function",
    "convert_pydantic_to_ernie_tool",
]
