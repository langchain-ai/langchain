"""LangChain OpenRouter integration."""

from importlib import metadata

from langchain_openrouter.chat_models import ChatOpenRouter
from langchain_openrouter.embeddings import OpenRouterEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatOpenRouter",
    "OpenRouterEmbeddings",
    "__version__",
]
