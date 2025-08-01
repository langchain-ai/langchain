"""Document extraction chains.

This module provides different strategies for extracting information from collections 
of documents using LangGraph and modern language models.

Available Strategies:
- Iterative: Processes documents sequentially, refining results at each step
- Map-Reduce: Processes documents in parallel, then combines results  
- Recursive: Hierarchical processing for documents
"""


# Strategy-specific functions
from langchain.chains.extraction.iterative import create_iterative_extractor
from langchain.chains.extraction.map_reduce import create_map_reduce_extractor
from langchain.chains.extraction.recursive import create_recursive_extractor

__all__ = [
    # Strategy-specific
    "create_iterative_extractor",
    "create_map_reduce_extractor", 
    "create_recursive_extractor",
]