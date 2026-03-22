"""Deprecated: use `langchain_fireworks._version` instead."""

import warnings

from langchain_fireworks._version import __version__  # noqa: F401

warnings.warn(
    "Importing from langchain_fireworks.version is deprecated. "
    "Use langchain_fireworks._version instead.",
    DeprecationWarning,
    stacklevel=2,
)
