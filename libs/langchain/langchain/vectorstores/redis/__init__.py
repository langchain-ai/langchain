from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import Redis
    from langchain_community.vectorstores.redis.base import RedisVectorStoreRetriever
    from langchain_community.vectorstores.redis.filters import (
        RedisFilter,
        RedisNum,
        RedisTag,
        RedisText,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "Redis": "langchain_community.vectorstores",
    "RedisFilter": "langchain_community.vectorstores.redis.filters",
    "RedisTag": "langchain_community.vectorstores.redis.filters",
    "RedisText": "langchain_community.vectorstores.redis.filters",
    "RedisNum": "langchain_community.vectorstores.redis.filters",
    "RedisVectorStoreRetriever": "langchain_community.vectorstores.redis.base",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "Redis",
    "RedisFilter",
    "RedisTag",
    "RedisText",
    "RedisNum",
    "RedisVectorStoreRetriever",
]
