from typing import Any

DEPRECATED_IMPORTS = [
    "EmbeddingsRedundantFilter",
    "EmbeddingsClusteringFilter",
    "_DocumentWithState",
    "get_stateful_documents",
    "_get_embeddings_from_stateful_docs",
    "_filter_similar_embeddings",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.document_transformers.embeddings_redundant_filter import {name}`"
        )

    raise AttributeError()
