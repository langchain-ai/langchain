from importlib import metadata

from __module_name__.chat_models import (  # type: ignore[import-not-found]
    Chat__ModuleName__,  # type: ignore[import-not-found]
)
from __module_name__.embeddings import (  # type: ignore[import-not-found]
    __ModuleName__Embeddings,  # type: ignore[import-not-found]
)
from __module_name__.llms import __ModuleName__LLM  # type: ignore[import-not-found]
from __module_name__.vectorstores import (  # type: ignore[import-not-found]
    __ModuleName__VectorStore,  # type: ignore[import-not-found]
)

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
