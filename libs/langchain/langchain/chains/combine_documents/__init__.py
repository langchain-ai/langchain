"""Different ways to combine documents."""

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    collapse_docs,
    split_list_of_docs,
)

__all__ = ["acollapse_docs", "collapse_docs", "split_list_of_docs"]
