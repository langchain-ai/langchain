from importlib import metadata

from langchain_needle.document_loaders import NeedleLoader
from langchain_needle.retrievers import NeedleRetriever

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "NeedleLoader",
    "NeedleRetriever",
    "__version__",
]
