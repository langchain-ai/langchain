"""LangChain integration for DocuWeave — layout-aware PDF chunking."""

from langchain_docuweave._version import __version__
from langchain_docuweave.document_loaders import DocuWeaveLoader

__all__ = [
    "DocuWeaveLoader",
    "__version__",
]
