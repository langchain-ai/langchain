"""``langchain-core`` defines the base abstractions for the LangChain ecosystem.

The interfaces for core components like chat models, LLMs, vector stores, retrievers,
and more are defined here. The universal invocation protocol (Runnables) along with
a syntax for combining components (LangChain Expression Language) are also defined here.

No third-party integrations are defined here. The dependencies are kept purposefully
very lightweight.
"""

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
