from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.storage import UpstashRedisByteStore, UpstashRedisStore

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "UpstashRedisStore": "langchain_community.storage",
    "UpstashRedisByteStore": "langchain_community.storage",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "UpstashRedisByteStore",
    "UpstashRedisStore",
]
