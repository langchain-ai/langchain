"""LangChain MiniMax integration."""

from importlib import metadata

from langchain_minimax.chat_models import ChatMiniMax

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatMiniMax",
    "__version__",
]
