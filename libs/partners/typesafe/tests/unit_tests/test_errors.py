"""Tests for TypeSafe provider and LangChain error classification."""

from __future__ import annotations

import pickle
from typing import Any

import httpx2
import pytest
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

from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
    parse_response,
)


@pytest.mark.parametrize(
    ("status", "provider_type", "langchain_type", "is_retryable"),
    [
        (400, TypeSafeBadRequestError, ModelInvalidRequestError, False),
        (401, TypeSafeAuthenticationError, ModelAuthenticationError, False),
        (403, TypeSafePermissionDeniedError, ModelPermissionDeniedError, False),
        (404, TypeSafeNotFoundError, ModelNotFoundError, False),
        (422, TypeSafeUnprocessableEntityError, ModelInvalidRequestError, False),
        (429, TypeSafeRateLimitError, ModelRateLimitError, True),
        (500, TypeSafeInternalServerError, ModelAPIError, True),
        (529, TypeSafeInternalServerError, ModelAPIError, True),
    ],
)
def test_status_errors_use_provider_and_langchain_types(
    status: int,
    provider_type: type[TypeSafeAPIError],
    langchain_type: type[ModelError],
    *,
    is_retryable: bool,
) -> None:
    """Each known status is catchable through provider and LangChain hierarchies."""
    request = httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone")
    response = httpx2.Response(status, json={"message": "failure"}, request=request)

    with pytest.raises(provider_type) as exc_info:
        parse_response(response)

    assert isinstance(exc_info.value, langchain_type)
    assert exc_info.value.is_retryable is is_retryable


def test_overloaded_error_has_safe_provider_description() -> None:
    """TypeSafe's nonstandard overloaded status remains useful without body text."""
    response = httpx2.Response(
        529,
        json={"message": "private overload detail"},
        request=httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone"),
    )

    with pytest.raises(TypeSafeInternalServerError) as exc_info:
        parse_response(response)

    assert "529 Overloaded" in str(exc_info.value)
    assert "private overload detail" not in str(exc_info.value)


def test_api_error_exposes_metadata_without_leaking_it_in_repr() -> None:
    """API errors expose structured context while keeping string forms sanitized."""
    request = httpx2.Request(
        "POST",
        "https://user:password@example.test/v1/systemone?token=secret#fragment",
    )
    response = httpx2.Response(
        400,
        json={"message": "private response detail"},
        headers={"x-typesafe-request-id": "req_123"},
        request=request,
    )

    with pytest.raises(TypeSafeBadRequestError) as exc_info:
        parse_response(response)

    error = exc_info.value
    assert error.status == 400
    assert error.status_code == 400
    assert error.body == {"message": "private response detail"}
    assert error.headers["x-typesafe-request-id"] == "req_123"
    assert error.request_id == "req_123"
    assert error.endpoint == "POST https://example.test/v1/systemone"
    assert "private response detail" not in str(error)
    assert "password" not in repr(error)
    assert "token=secret" not in repr(error)


def test_endpoint_sanitization_preserves_ipv6_and_port() -> None:
    """Sanitization retains IPv6 addressing and explicit ports."""
    request = httpx2.Request(
        "POST",
        "https://[2001:db8::1]:8443/v1/systemone?token=secret",
    )
    response = httpx2.Response(400, request=request)

    with pytest.raises(TypeSafeBadRequestError) as exc_info:
        parse_response(response)

    assert exc_info.value.endpoint == "POST https://[2001:db8::1]:8443/v1/systemone"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "125"}, 125.0),
        ({"retry-after": "2"}, 2000.0),
        ({"retry-after-ms": "bad", "retry-after": "3"}, 3000.0),
        ({"retry-after-ms": "bad", "retry-after": "bad"}, None),
    ],
)
def test_rate_limit_error_parses_retry_delay(
    headers: dict[str, str], expected: float | None
) -> None:
    """Rate-limit responses expose the server-requested delay in milliseconds."""
    response = httpx2.Response(
        429,
        headers=headers,
        request=httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone"),
    )

    with pytest.raises(TypeSafeRateLimitError) as exc_info:
        parse_response(response)

    assert exc_info.value.retry_after_ms == expected


def test_response_validation_error_reports_field_path() -> None:
    """Malformed successful responses identify the first invalid field."""
    body: dict[str, Any] = {"model": "jev-latest", "answers": []}
    response = httpx2.Response(
        200,
        json=body,
        request=httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone"),
    )

    with pytest.raises(TypeSafeAPIResponseValidationError) as exc_info:
        parse_response(response)

    error = exc_info.value
    assert error.status == 200
    assert error.body == body
    assert error.field_path == "answers"


def test_connection_error_uses_standard_hierarchies() -> None:
    """Connection failures are catchable as provider, LangChain, and Python errors."""
    error = TypeSafeAPIConnectionError("Unable to connect")

    assert isinstance(error, ModelConnectionError)
    assert isinstance(error, ConnectionError)
    assert error.is_retryable is True


def test_timeout_error_uses_standard_hierarchies() -> None:
    """Timeouts retain their setting and all provider and standard base types."""
    timeout = httpx2.Timeout(10.0)
    error = TypeSafeAPITimeoutError(timeout)

    assert isinstance(error, TypeSafeAPIConnectionError)
    assert isinstance(error, ModelTimeoutError)
    assert isinstance(error, TimeoutError)
    assert error.is_retryable is True
    assert error.timeout is timeout


@pytest.mark.parametrize(
    "error",
    [
        TypeSafeAuthenticationError(401, {}, httpx2.Headers()),
        TypeSafeRateLimitError(
            429,
            {},
            httpx2.Headers({"retry-after-ms": "125"}),
        ),
        TypeSafeAPITimeoutError(10.0),
        TypeSafeAPIResponseValidationError(
            200,
            {},
            httpx2.Headers(),
            "answers.urgent.noul",
        ),
    ],
    ids=lambda error: type(error).__name__,
)
def test_errors_round_trip_through_pickle(error: Exception) -> None:
    """Structured errors retain their type and attributes across process boundaries."""
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert type(restored) is type(error)
    assert restored.args == error.args
    assert vars(restored) == vars(error)
