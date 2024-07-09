from importlib import metadata

from __module_name__.chat_models import Chat__ModuleName__
from __module_name__.embeddings import __ModuleName__Embeddings
from __module_name__.llms import __ModuleName__LLM
from __module_name__.vectorstores import __ModuleName__VectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "Chat__ModuleName__",
    "__ModuleName__LLM",
    "__ModuleName__VectorStore",
    "__ModuleName__Embeddings",
    "__version__",
]
