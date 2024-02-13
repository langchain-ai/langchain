"""Main entrypoint into package."""
from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

# Check that we have gigachain_core package instead of langchain_core
from langchain_core import __gigachain_core
