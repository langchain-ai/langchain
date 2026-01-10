"""This module provides dynamic access to deprecated JSON tools in LangChain.

It ensures backward compatibility by forwarding references such as
`JsonGetValueTool`, `JsonListKeysTool`, and `JsonSpec` to their updated
locations within the `langchain_community.tools` namespace.

This setup allows legacy code to continue working while guiding developers
toward using the updated module paths.
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import JsonGetValueTool, JsonListKeysTool
    from langchain_community.tools.json.tool import JsonSpec

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "JsonSpec": "langchain_community.tools.json.tool",
    "JsonListKeysTool": "langchain_community.tools",
    "JsonGetValueTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Dynamically retrieve attributes from the updated module path.

    This method is used to resolve deprecated attribute imports
    at runtime and forward them to their new locations.

    Args:
        name: The name of the attribute to import.

    Returns:
        The resolved attribute from the appropriate updated module.
    """
    return _import_attribute(name)


__all__ = [
    "JsonGetValueTool",
    "JsonListKeysTool",
    "JsonSpec",
]
