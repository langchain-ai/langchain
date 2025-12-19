from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.powerbi.prompt import (
        POWERBI_CHAT_PREFIX,
        POWERBI_CHAT_SUFFIX,
        POWERBI_PREFIX,
        POWERBI_SUFFIX,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "POWERBI_CHAT_PREFIX": "langchain_community.agent_toolkits.powerbi.prompt",
    "POWERBI_CHAT_SUFFIX": "langchain_community.agent_toolkits.powerbi.prompt",
    "POWERBI_PREFIX": "langchain_community.agent_toolkits.powerbi.prompt",
    "POWERBI_SUFFIX": "langchain_community.agent_toolkits.powerbi.prompt",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "POWERBI_CHAT_PREFIX",
    "POWERBI_CHAT_SUFFIX",
    "POWERBI_PREFIX",
    "POWERBI_SUFFIX",
]
