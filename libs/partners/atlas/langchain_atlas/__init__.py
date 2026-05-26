"""LangChain Atlas integration."""

from importlib import metadata

from langchain_atlas.chat_models import ChatAtlas

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "ChatAtlas",
    "__version__",
]
