"""Pair TypeSafe SDK exceptions with LangChain's standard model errors.

The TypeSafe SDK already raises a complete, well-specified exception hierarchy with
status, body, headers, endpoint, request ID, and retry metadata. The only thing it
cannot know about is LangChain's provider-independent `ModelError` hierarchy, which
lets callers handle a rate limit or an authentication failure without importing a
provider package.

This module defines one subclass per SDK exception that inherits from both the SDK
class and the matching `langchain_core.exceptions` class, then re-raises SDK errors as
their paired subclass. Because every subclass derives from the original SDK class,
`except typesafe_sdk.TypeSafeRateLimitError` keeps working unchanged alongside
`except ModelRateLimitError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import typesafe_sdk as ts
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)


class TypeSafeBadRequestError(ts.TypeSafeBadRequestError, ModelInvalidRequestError):
    """TypeSafe rejected the request as invalid (HTTP 400)."""


class TypeSafeAuthenticationError(
    ts.TypeSafeAuthenticationError,
    ModelAuthenticationError,
):
    """Authentication with TypeSafe failed (HTTP 401)."""


class TypeSafePermissionDeniedError(
    ts.TypeSafePermissionDeniedError,
    ModelPermissionDeniedError,
):
    """The TypeSafe credential cannot perform the request (HTTP 403)."""


class TypeSafeNotFoundError(ts.TypeSafeNotFoundError, ModelNotFoundError):
    """The requested TypeSafe resource or model was not found (HTTP 404)."""


class TypeSafeUnprocessableEntityError(
    ts.TypeSafeUnprocessableEntityError,
    ModelInvalidRequestError,
):
    """TypeSafe rejected the request body during validation (HTTP 422)."""


class TypeSafeRateLimitError(ts.TypeSafeRateLimitError, ModelRateLimitError):
    """The TypeSafe rate limit was exceeded (HTTP 429)."""


class TypeSafeInternalServerError(ts.TypeSafeInternalServerError, ModelAPIError):
    """TypeSafe failed to process the request (HTTP 5xx, including 529)."""


class TypeSafeAPIResponseValidationError(
    ts.TypeSafeAPIResponseValidationError,
    ModelAPIError,
):
    """A successful TypeSafe response was missing or contained invalid data."""


class TypeSafeAPIConnectionError(ts.TypeSafeAPIConnectionError, ModelConnectionError):
    """A TypeSafe request failed without receiving an HTTP response."""


class TypeSafeAPITimeoutError(
    ts.TypeSafeAPITimeoutError,
    TypeSafeAPIConnectionError,
    ModelTimeoutError,
):
    """A TypeSafe request exceeded its configured timeout.

    Subclassing this package's `TypeSafeAPIConnectionError` mirrors the SDK hierarchy,
    so `except TypeSafeAPIConnectionError` catches timeouts from either import.
    """


_API_ERRORS: dict[type[ts.TypeSafeAPIError], type[ts.TypeSafeAPIError]] = {
    ts.TypeSafeBadRequestError: TypeSafeBadRequestError,
    ts.TypeSafeAuthenticationError: TypeSafeAuthenticationError,
    ts.TypeSafePermissionDeniedError: TypeSafePermissionDeniedError,
    ts.TypeSafeNotFoundError: TypeSafeNotFoundError,
    ts.TypeSafeUnprocessableEntityError: TypeSafeUnprocessableEntityError,
    ts.TypeSafeRateLimitError: TypeSafeRateLimitError,
    ts.TypeSafeInternalServerError: TypeSafeInternalServerError,
}


@contextmanager
def with_standard_errors() -> Iterator[None]:
    """Re-raise TypeSafe SDK exceptions as their LangChain-aware subclasses.

    Exceptions that are already paired subclasses pass through untouched, so nesting
    this context manager does not re-wrap an error. SDK exception types without a
    LangChain counterpart also pass through unchanged.

    Yields:
        `None`. The enclosed block performs the TypeSafe request.

    Raises:
        ModelError: A paired subclass that is both the original TypeSafe SDK exception
            type and the corresponding `langchain_core.exceptions` type.
    """
    try:
        yield
    except ModelError:
        # Already translated; re-wrapping would discard the original traceback.
        raise
    except ts.TypeSafeAPITimeoutError as error:
        raise TypeSafeAPITimeoutError(error.timeout) from error
    except ts.TypeSafeAPIConnectionError as error:
        raise TypeSafeAPIConnectionError(*error.args) from error
    except ts.TypeSafeAPIResponseValidationError as error:
        raise TypeSafeAPIResponseValidationError(
            error.status,
            error.body,
            error.headers,
            error.field_path,
            error.endpoint,
        ) from error
    except ts.TypeSafeAPIError as error:
        paired = _API_ERRORS.get(type(error))
        if paired is None:
            raise
        raise paired(
            error.status,
            error.body,
            error.headers,
            endpoint=error.endpoint,
        ) from error


__all__ = [
    "TypeSafeAPIConnectionError",
    "TypeSafeAPIResponseValidationError",
    "TypeSafeAPITimeoutError",
    "TypeSafeAuthenticationError",
    "TypeSafeBadRequestError",
    "TypeSafeInternalServerError",
    "TypeSafeNotFoundError",
    "TypeSafePermissionDeniedError",
    "TypeSafeRateLimitError",
    "TypeSafeUnprocessableEntityError",
    "with_standard_errors",
]
