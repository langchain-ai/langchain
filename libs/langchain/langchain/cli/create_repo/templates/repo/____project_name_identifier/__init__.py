"""____project_name_identifier package."""
from importlib import metadata

from ____project_name_identifier.chain import get_chain

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [__version__, "get_chain"]
