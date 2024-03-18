from typing import Any

DEPRECATED_IMPORTS = [
    "RedisFilterOperator",
    "RedisFilter",
    "RedisFilterField",
    "check_operator_misuse",
    "RedisTag",
    "RedisNum",
    "RedisText",
    "RedisFilterExpression",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.vectorstores.redis.filters import {name}`"
        )

    raise AttributeError()
