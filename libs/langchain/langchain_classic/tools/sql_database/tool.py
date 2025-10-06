from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        BaseSQLDatabaseTool,
        InfoSQLDatabaseTool,
        ListSQLDatabaseTool,
        QuerySQLCheckerTool,
        QuerySQLDataBaseTool,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "BaseSQLDatabaseTool": "langchain_community.tools",
    "QuerySQLDataBaseTool": "langchain_community.tools",
    "InfoSQLDatabaseTool": "langchain_community.tools",
    "ListSQLDatabaseTool": "langchain_community.tools",
    "QuerySQLCheckerTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BaseSQLDatabaseTool",
    "InfoSQLDatabaseTool",
    "ListSQLDatabaseTool",
    "QuerySQLCheckerTool",
    "QuerySQLDataBaseTool",
]
