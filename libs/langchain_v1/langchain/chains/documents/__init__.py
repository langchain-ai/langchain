"""Document extraction chains.

This module provides different strategies for extracting information from collections
of documents using LangGraph and modern language models.

Available Strategies:
- Stuff: Processes all documents together in a single context window
- Map-Reduce: Processes documents in parallel (map), then combines results (reduce)
"""

from langchain.chains.documents.map_reduce import create_map_reduce_chain
from langchain.chains.documents.stuff import create_stuff_documents_chain

__all__ = [
    "create_map_reduce_chain",
    "create_stuff_documents_chain",
]
