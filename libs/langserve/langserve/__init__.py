"""Main entrypoint into package."""
from importlib import metadata

from langserve.client import RemoteRunnable
from langserve.server import add_routes

__all__ = ["RemoteRunnable", "add_routes"]


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)
