from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utilities.redis import (
        TokenEscaper,
        check_redis_module_exist,
        get_client,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "TokenEscaper": "langchain_community.utilities.redis",
    "check_redis_module_exist": "langchain_community.utilities.redis",
    "get_client": "langchain_community.utilities.redis",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "TokenEscaper",
    "check_redis_module_exist",
    "get_client",
]
