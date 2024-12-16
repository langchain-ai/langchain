from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.cache import (
        AstraDBCache,
        AstraDBSemanticCache,
        AzureCosmosDBSemanticCache,
        CassandraCache,
        CassandraSemanticCache,
        FullLLMCache,
        FullMd5LLMCache,
        GPTCache,
        InMemoryCache,
        MomentoCache,
        RedisCache,
        RedisSemanticCache,
        SQLAlchemyCache,
        SQLAlchemyMd5Cache,
        SQLiteCache,
        UpstashRedisCache,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "FullLLMCache": "langchain_community.cache",
    "SQLAlchemyCache": "langchain_community.cache",
    "SQLiteCache": "langchain_community.cache",
    "UpstashRedisCache": "langchain_community.cache",
    "RedisCache": "langchain_community.cache",
    "RedisSemanticCache": "langchain_community.cache",
    "GPTCache": "langchain_community.cache",
    "MomentoCache": "langchain_community.cache",
    "InMemoryCache": "langchain_community.cache",
    "CassandraCache": "langchain_community.cache",
    "CassandraSemanticCache": "langchain_community.cache",
    "FullMd5LLMCache": "langchain_community.cache",
    "SQLAlchemyMd5Cache": "langchain_community.cache",
    "AstraDBCache": "langchain_community.cache",
    "AstraDBSemanticCache": "langchain_community.cache",
    "AzureCosmosDBSemanticCache": "langchain_community.cache",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "FullLLMCache",
    "SQLAlchemyCache",
    "SQLiteCache",
    "UpstashRedisCache",
    "RedisCache",
    "RedisSemanticCache",
    "GPTCache",
    "MomentoCache",
    "InMemoryCache",
    "CassandraCache",
    "CassandraSemanticCache",
    "FullMd5LLMCache",
    "SQLAlchemyMd5Cache",
    "AstraDBCache",
    "AstraDBSemanticCache",
    "AzureCosmosDBSemanticCache",
]
