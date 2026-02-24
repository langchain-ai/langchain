"""Retrievers.

!!! warning "Modules moved"

    With the release of `langchain 1.0.0`, most retriever implementations were moved
    to `langchain-classic` or dedicated partner packages. This module provides
    backwards-compatible re-exports of the core retriever abstractions from
    `langchain-core`.
"""

from langchain_core.retrievers import BaseRetriever

__all__ = ["BaseRetriever"]
