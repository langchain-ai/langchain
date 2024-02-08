"""Different ways to combine documents."""

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    collapse_docs,
    split_list_of_docs,
)
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

__all__ = [
    "acollapse_docs",
    "collapse_docs",
    "split_list_of_docs",
    "create_stuff_documents_chain",
]
