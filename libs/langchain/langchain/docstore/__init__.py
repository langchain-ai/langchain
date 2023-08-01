"""Wrappers on top of docstores."""
from langchain.docstore.arbitrary_fn import DocstoreFn
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.wikipedia import Wikipedia

__all__ = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]
