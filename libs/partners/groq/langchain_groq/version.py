"""Deprecated: use `langchain_groq._version` instead."""

import warnings

from langchain_groq._version import __version__  # noqa: F401

warnings.warn(
    "Importing from langchain_groq.version is deprecated. "
    "Use langchain_groq._version instead.",
    DeprecationWarning,
    stacklevel=2,
)
