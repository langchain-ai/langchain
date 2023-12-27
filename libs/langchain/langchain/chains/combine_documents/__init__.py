"""Different ways to combine documents."""

from langchain.chains.combine_documents.map_reduce import (
    create_map_documents_chain,
    create_map_reduce_documents_chain,
)
from langchain.chains.combine_documents.map_rerank import (
    create_map_rerank_documents_chain,
)
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    collapse_docs,
    create_collapse_documents_chain,
    split_list_of_docs,
)
from langchain.chains.combine_documents.refine import create_refine_documents_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

__all__ = [
    "acollapse_docs",
    "collapse_docs",
    "split_list_of_docs",
    "create_stuff_documents_chain",
    "create_map_documents_chain",
    "create_map_rerank_documents_chain",
    "create_map_reduce_documents_chain",
    "create_refine_documents_chain",
]
