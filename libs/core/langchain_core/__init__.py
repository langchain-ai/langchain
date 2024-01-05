from importlib import metadata

from langchain_core._api import (
    surface_langchain_beta_warnings,
    surface_langchain_deprecation_warnings,
)
from langchain_core.utils.interactive_env import is_interactive_env

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

if not is_interactive_env():
    surface_langchain_deprecation_warnings()
    surface_langchain_beta_warnings()

surface_langchain_deprecation_warnings()
surface_langchain_beta_warnings()
