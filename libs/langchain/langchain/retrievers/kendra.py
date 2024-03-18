from typing import Any

DEPRECATED_IMPORTS = [
    "clean_excerpt",
    "combined_text",
    "DocumentAttributeValueType",
    "Highlight",
    "TextWithHighLights",
    "AdditionalResultAttributeValue",
    "AdditionalResultAttribute",
    "DocumentAttributeValue",
    "DocumentAttribute",
    "ResultItem",
    "QueryResultItem",
    "RetrieveResultItem",
    "QueryResult",
    "RetrieveResult",
    "AmazonKendraRetriever",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.retrievers.kendra import {name}`"
        )

    raise AttributeError()
