"""``langchain-core`` defines abstractions for common components and a syntax for combining them.

The interfaces for core components like chat models, LLMs, vector stores, retrievers,
and more are defined here. No third-party integrations are defined here. The
dependencies are kept purposefully very lightweight.
"""  # noqa: E501

from importlib import metadata

from langchain_core._api import (
    surface_langchain_beta_warnings,
    surface_langchain_deprecation_warnings,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

surface_langchain_deprecation_warnings()
surface_langchain_beta_warnings()
