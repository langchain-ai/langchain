"""This module provides dynamic access to deprecated Jira tools.

When attributes like `JiraAction` are accessed, they are redirected to their new
locations in `langchain_community.tools`. This ensures backward compatibility
while warning developers about deprecation.

Attributes:
    JiraAction (deprecated): Dynamically loaded from langchain_community.tools.
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import JiraAction

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {"JiraAction": "langchain_community.tools"}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Dynamically retrieve attributes from the updated module path.

    Args:
        name: The name of the attribute to import.

    Returns:
        The resolved attribute from the updated path.
    """
    return _import_attribute(name)


__all__ = [
    "JiraAction",
]
