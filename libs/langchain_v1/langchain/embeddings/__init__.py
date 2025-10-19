"""Embeddings.

!!! warning "Reference docs"
    This page contains **reference documentation** for Embeddings. See
    [the docs](https://docs.langchain.com/oss/python/langchain/retrieval#embedding-models)
    for conceptual guides, tutorials, and examples on using Embeddings.
"""

from langchain_core.embeddings import Embeddings

from langchain.embeddings.base import init_embeddings

__all__ = [
    "Embeddings",
    "init_embeddings",
]
