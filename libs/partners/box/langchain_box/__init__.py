from importlib import metadata

from langchain_box.document_loaders import BoxLoader
from langchain_box.retrievers import BoxRetriever
from langchain_box.utilities import BoxAuth, BoxAuthType, _BoxAPIWrapper

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
    "_BoxAPIWrapper",
    "__version__",
]
