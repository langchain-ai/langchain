from importlib import metadata

from langchain_box.document_loaders import BoxLoader
from langchain_box.retrievers import BoxRetriever
from langchain_box.utilities.box import (
    BoxAuth,
    BoxAuthType,
    BoxSearchOptions,
    DocumentFiles,
    SearchTypeFilter,
    _BoxAPIWrapper,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "BoxLoader",
    "BoxRetriever",
    "BoxAuth",
    "BoxAuthType",
    "BoxSearchOptions",
    "DocumentFiles",
    "SearchTypeFilter",
    "_BoxAPIWrapper",
    "__version__",
]
