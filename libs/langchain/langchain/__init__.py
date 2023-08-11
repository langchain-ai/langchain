"""Main entrypoint into package."""

from importlib import metadata
from typing import Any, Optional

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

verbose: bool = False
debug: bool = False
# Type as `Any` to avoid importing Cache
llm_cache: Optional[Any] = None
