"""Main entrypoint into package."""

import warnings
from importlib import metadata

warnings.warn(
    "The weaviate related classes in the langchain_community package are deprecated. "
    "Please download and install the langchain-weaviate package.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)
