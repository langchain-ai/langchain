from typing import Any

DEPRECATED_IMPORTS = [
    "create_retry_decorator",
    "raise_vertex_import_error",
    "init_vertexai",
    "get_client_info",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.utilities.vertexai import {name}`"
        )

    raise AttributeError()
