from langchain._api import create_importer

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from langchain_community.vectorstores.redis.filters import RedisFilterOperator
    from langchain_community.vectorstores.redis.filters import RedisFilter
    from langchain_community.vectorstores.redis.filters import RedisFilterField
    from langchain_community.vectorstores.redis.filters import check_operator_misuse
    from langchain_community.vectorstores.redis.filters import RedisTag
    from langchain_community.vectorstores.redis.filters import RedisNum
    from langchain_community.vectorstores.redis.filters import RedisText
    from langchain_community.vectorstores.redis.filters import RedisFilterExpression
            
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {"RedisFilterOperator": "langchain_community.vectorstores.redis.filters", "RedisFilter": "langchain_community.vectorstores.redis.filters", "RedisFilterField": "langchain_community.vectorstores.redis.filters", "check_operator_misuse": "langchain_community.vectorstores.redis.filters", "RedisTag": "langchain_community.vectorstores.redis.filters", "RedisNum": "langchain_community.vectorstores.redis.filters", "RedisText": "langchain_community.vectorstores.redis.filters", "RedisFilterExpression": "langchain_community.vectorstores.redis.filters"}
        
_import_attribute=create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
__all__ = ["RedisFilterOperator",
"RedisFilter",
"RedisFilterField",
"check_operator_misuse",
"RedisTag",
"RedisNum",
"RedisText",
"RedisFilterExpression",
]
