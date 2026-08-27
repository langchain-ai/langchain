"""This module provides dynamic access to deprecated Zapier tools in LangChain.

It supports backward compatibility by forwarding references such as
`ZapierNLAListActions` and `ZapierNLARunAction` to their updated locations
in the `langchain_community.tools` package.

Developers using older import paths will continue to function, while LangChain
internally redirects access to the newer, supported module structure.
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import ZapierNLAListActions, ZapierNLARunAction

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ZapierNLARunAction": "langchain_community.tools",
    "ZapierNLAListActions": "langchain_community.tools",
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
    "ZapierNLAListActions",
    "ZapierNLARunAction",
]
