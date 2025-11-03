"""Embeddings models.

!!! warning "Reference docs"
    This page contains **reference documentation** for Embeddings. See
    [the docs](https://docs.langchain.com/oss/python/langchain/retrieval#embedding-models)
    for conceptual guides, tutorials, and examples on using Embeddings.

!!! warning "Modules moved"
    With the release of `langchain 1.0.0`, several embeddings modules were moved to
    `langchain-classic`, such as `CacheBackedEmbeddings` and all community
    embeddings. See [list](https://github.com/langchain-ai/langchain/blob/bdf1cd383ce36dc18381a3bf3fb0a579337a32b5/libs/langchain/langchain/embeddings/__init__.py)
    of moved modules to inform your migration.
"""

from langchain_core.embeddings import Embeddings

from langchain.embeddings.base import init_embeddings

__all__ = [
    "Embeddings",
    "init_embeddings",
]
