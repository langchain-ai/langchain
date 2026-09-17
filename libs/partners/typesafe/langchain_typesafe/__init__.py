"""LangChain integration for TypeSafe classifiers.

This package adds a LangChain `Runnable` on top of the official TypeSafe Python SDK.
The SDK owns the wire protocol, question and answer types, retries, and response
validation; this package adds tracing, LangChain message handling, and errors that
also subclass LangChain's standard `ModelError` hierarchy.

Question types are re-exported here for convenience. Answer and response types are
not: they arrive on the response object and can be imported from `typesafe_sdk` when
an explicit annotation is needed.
"""

from typesafe_sdk import (
    Choice,
    Noul,
    NoulCriteria,
    RetryPolicy,
    Score,
    TypeSafeAPIError,
    TypeSafeError,
)

from langchain_typesafe._errors import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)
from langchain_typesafe._version import __version__
from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import State

__all__ = [
    "Choice",
    "Noul",
    "NoulCriteria",
    "RetryPolicy",
    "Score",
    "State",
    "TypeSafeAPIConnectionError",
    "TypeSafeAPIError",
    "TypeSafeAPIResponseValidationError",
    "TypeSafeAPITimeoutError",
    "TypeSafeAuthenticationError",
    "TypeSafeBadRequestError",
    "TypeSafeClassifier",
    "TypeSafeError",
    "TypeSafeInternalServerError",
    "TypeSafeNotFoundError",
    "TypeSafePermissionDeniedError",
    "TypeSafeRateLimitError",
    "TypeSafeUnprocessableEntityError",
    "__version__",
]
