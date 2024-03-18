from typing import Any

DEPRECATED_IMPORTS = [
    "POWERBI_PREFIX",
    "POWERBI_SUFFIX",
    "POWERBI_CHAT_PREFIX",
    "POWERBI_CHAT_SUFFIX",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.agent_toolkits.powerbi.prompt import {name}`"
        )

    raise AttributeError()
