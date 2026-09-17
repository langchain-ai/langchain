"""Unit tests for TypeSafe error translation."""

from __future__ import annotations

import httpx2
import pytest
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
    with_standard_errors,
)

# Each SDK HTTP exception, the status the provider reports it with, the package
# exception it is translated to, and the standard LangChain error it also becomes.
_API_CASES = [
    (
        ts.TypeSafeBadRequestError,
        400,
        TypeSafeBadRequestError,
        ModelInvalidRequestError,
    ),
    (
        ts.TypeSafeAuthenticationError,
        401,
        TypeSafeAuthenticationError,
        ModelAuthenticationError,
    ),
    (
        ts.TypeSafePermissionDeniedError,
        403,
        TypeSafePermissionDeniedError,
        ModelPermissionDeniedError,
    ),
    (ts.TypeSafeNotFoundError, 404, TypeSafeNotFoundError, ModelNotFoundError),
    (
        ts.TypeSafeUnprocessableEntityError,
        422,
        TypeSafeUnprocessableEntityError,
        ModelInvalidRequestError,
    ),
    (ts.TypeSafeRateLimitError, 429, TypeSafeRateLimitError, ModelRateLimitError),
    (ts.TypeSafeInternalServerError, 500, TypeSafeInternalServerError, ModelAPIError),
    (ts.TypeSafeInternalServerError, 529, TypeSafeInternalServerError, ModelAPIError),
]

_REQUEST = httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone")


def _raise(
    sdk_error: type[ts.TypeSafeAPIError],
    status: int,
    *,
    body: object = None,
    headers: httpx2.Headers | None = None,
) -> None:
    """Raise an SDK HTTP error through the translator."""
    with with_standard_errors():
        raise sdk_error(
            status,
            body,
            headers if headers is not None else httpx2.Headers(),
            endpoint=f"POST {_REQUEST.url}",
        )


@pytest.mark.parametrize(("sdk_error", "status", "expected", "standard"), _API_CASES)
def test_status_maps_to_paired_error(
    sdk_error: type[ts.TypeSafeAPIError],
    status: int,
    expected: type[Exception],
    standard: type[ModelError],
) -> None:
    """Each classified status raises a package error that is also a `ModelError`."""
    with pytest.raises(expected) as exc_info:
        _raise(sdk_error, status)

    error = exc_info.value
    assert isinstance(error, standard)
    assert isinstance(error, sdk_error)
    assert isinstance(error, ts.TypeSafeError)
    assert error.status == status  # type: ignore[attr-defined]


@pytest.mark.parametrize(("sdk_error", "status", "expected", "standard"), _API_CASES)
def test_sdk_exception_types_still_catch_paired_errors(
    sdk_error: type[ts.TypeSafeAPIError],
    status: int,
    expected: type[Exception],
    standard: type[ModelError],
) -> None:
    """Code that catches the SDK's own exception types keeps working."""
    del expected, standard
    with pytest.raises(sdk_error):
        _raise(sdk_error, status)


def test_retryable_flags_follow_the_condition() -> None:
    """Retryability matches LangChain's standard classification."""
    with pytest.raises(TypeSafeRateLimitError) as rate_limited:
        _raise(ts.TypeSafeRateLimitError, 429)
    with pytest.raises(TypeSafeInternalServerError) as overloaded:
        _raise(ts.TypeSafeInternalServerError, 529)
    with pytest.raises(TypeSafeAuthenticationError) as unauthenticated:
        _raise(ts.TypeSafeAuthenticationError, 401)

    assert rate_limited.value.is_retryable
    assert overloaded.value.is_retryable
    assert not unauthenticated.value.is_retryable


def test_translation_preserves_response_metadata() -> None:
    """Status, body, headers, request ID, and endpoint survive translation."""
    body = {"error": "model overloaded"}
    headers = httpx2.Headers({"x-typesafe-request-id": "req_123"})

    with pytest.raises(TypeSafeInternalServerError) as exc_info:
        _raise(ts.TypeSafeInternalServerError, 529, body=body, headers=headers)

    error = exc_info.value
    assert error.status == 529
    assert error.body == body
    assert error.request_id == "req_123"
    assert error.headers["x-typesafe-request-id"] == "req_123"
    assert error.endpoint == f"POST {_REQUEST.url}"


def test_translation_preserves_the_message() -> None:
    """The translated error renders the same string as the SDK error."""
    original = ts.TypeSafeBadRequestError(
        400,
        {"error": "criteria must not be empty"},
        httpx2.Headers(),
        endpoint=f"POST {_REQUEST.url}",
    )

    with pytest.raises(TypeSafeBadRequestError) as exc_info, with_standard_errors():
        raise original

    assert str(exc_info.value) == str(original)


def test_rate_limit_error_keeps_retry_delay() -> None:
    """A 429 retains the provider's requested retry delay."""
    with pytest.raises(TypeSafeRateLimitError) as exc_info:
        _raise(
            ts.TypeSafeRateLimitError,
            429,
            headers=httpx2.Headers({"retry-after-ms": "1500"}),
        )

    assert exc_info.value.retry_after_ms == 1500


def test_response_validation_error_keeps_field_path() -> None:
    """A malformed successful response reports the offending field."""
    with (
        pytest.raises(TypeSafeAPIResponseValidationError) as exc_info,
        with_standard_errors(),
    ):
        raise ts.TypeSafeAPIResponseValidationError(
            200,
            {"answers": {}},
            httpx2.Headers(),
            "answers.tone.confidence",
            "POST https://api.typesafe.ai/v1/systemone",
        )

    assert exc_info.value.field_path == "answers.tone.confidence"
    assert isinstance(exc_info.value, ModelAPIError)


def test_timeout_error_is_paired() -> None:
    """A timeout is both an SDK timeout and a LangChain timeout."""
    with pytest.raises(TypeSafeAPITimeoutError) as exc_info, with_standard_errors():
        raise ts.TypeSafeAPITimeoutError(7.5)

    error = exc_info.value
    assert error.timeout == 7.5
    assert isinstance(error, ModelTimeoutError)
    assert isinstance(error, TimeoutError)
    assert isinstance(error, TypeSafeAPIConnectionError)
    assert error.is_retryable


def test_connection_error_is_paired() -> None:
    """A connection failure is both an SDK and a LangChain connection error."""
    msg = "Unable to connect to the TypeSafe API."
    with (
        pytest.raises(TypeSafeAPIConnectionError) as exc_info,
        with_standard_errors(),
    ):
        raise ts.TypeSafeAPIConnectionError(msg)

    error = exc_info.value
    assert isinstance(error, ModelConnectionError)
    assert isinstance(error, ConnectionError)
    assert error.is_retryable
    assert "Unable to connect" in str(error)


def test_already_translated_errors_pass_through() -> None:
    """Nesting the translator does not re-wrap or lose the original traceback."""
    with (
        pytest.raises(TypeSafeRateLimitError) as exc_info,
        with_standard_errors(),  # outer
        with_standard_errors(),  # inner
    ):
        _raise(ts.TypeSafeRateLimitError, 429)

    assert exc_info.value.__cause__ is not None
    assert not isinstance(exc_info.value.__cause__, ModelError)


def test_unclassified_status_keeps_the_sdk_error() -> None:
    """A status the SDK does not classify is re-raised unchanged."""
    with pytest.raises(ts.TypeSafeAPIError) as exc_info:
        _raise(ts.TypeSafeAPIError, 418)

    assert not isinstance(exc_info.value, ModelError)
    assert exc_info.value.status == 418


def test_successful_blocks_are_untouched() -> None:
    """The translator returns control normally when nothing is raised."""
    with with_standard_errors():
        value = 1 + 1

    assert value == 2
