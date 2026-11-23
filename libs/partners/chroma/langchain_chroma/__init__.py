"""LangChain integration for Chroma vector database."""

from langchain_chroma._version import __version__
from langchain_chroma.vectorstores import Chroma

__all__ = [
    "Chroma",
    "__version__",
]
