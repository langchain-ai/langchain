from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.sql.prompt import (
        SQL_FUNCTIONS_SUFFIX,
        SQL_PREFIX,
        SQL_SUFFIX,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "SQL_PREFIX": "langchain_community.agent_toolkits.sql.prompt",
    "SQL_SUFFIX": "langchain_community.agent_toolkits.sql.prompt",
    "SQL_FUNCTIONS_SUFFIX": "langchain_community.agent_toolkits.sql.prompt",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = ["SQL_FUNCTIONS_SUFFIX", "SQL_PREFIX", "SQL_SUFFIX"]
