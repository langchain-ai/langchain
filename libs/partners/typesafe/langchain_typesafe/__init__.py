"""LangChain integration for TypeSafe classifiers.

This package adds a LangChain `Runnable` on top of the official TypeSafe Python SDK.
The SDK owns the wire protocol, question and answer types, retries, and response
validation; this package adds tracing, LangChain message handling, and errors that
also subclass LangChain's standard `ModelError` hierarchy.

Only the names needed to build a classifier are exported. Question types come from the
SDK and are re-exported here because every caller needs them. Answers and responses
arrive on the response object, and errors are catchable through either
`typesafe_sdk` or `langchain_core.exceptions`, so neither is mirrored here.
"""

from typesafe_sdk import Choice, Noul, Score

from langchain_typesafe._version import __version__
from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import State

__all__ = [
    "Choice",
    "Noul",
    "Score",
    "State",
    "TypeSafeClassifier",
    "__version__",
]
