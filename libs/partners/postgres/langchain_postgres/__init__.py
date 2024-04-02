from importlib import metadata

from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [
    "__version__",
    "PostgresChatMessageHistory",
]
