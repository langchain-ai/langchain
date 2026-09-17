"""HTTP response handling and errors for the TypeSafe integration."""

from __future__ import annotations

import math
import time
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from pydantic import ValidationError
from typing_extensions import override

from langchain_typesafe.types import ClassificationResponse

_REQUEST_ID_HEADER = "x-typesafe-request-id"
_RETRY_AFTER_HEADER = "retry-after"
_RETRY_AFTER_MS_HEADER = "retry-after-ms"
_SAFE_STATUS_MESSAGES = {529: "Overloaded"}


class TypeSafeError(Exception):
    """Base exception for errors raised by the TypeSafe integration."""


class TypeSafeAPIError(TypeSafeError):
    """An unsuccessful HTTP response with its body and request metadata.

    Attributes:
        status: HTTP response status code.
        body: Parsed JSON error body, plain response text, or `None`.
        headers: HTTP response headers.
        endpoint: Request method and URL without credentials, query, or fragment.
        request_id: Value of the `x-typesafe-request-id` response header.

    The body and headers are available for programmatic error handling but deliberately
    excluded from `str` and `repr` to avoid exposing classified state or credentials in
    logs and tracebacks.
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx2.Headers,
        message: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Create an API error from a TypeSafe HTTP response.

        Args:
            status: HTTP response status code.
            body: Parsed JSON body, plain response text, or `None`.
            headers: HTTP response headers.
            message: Optional safe message override that does not contain response data.
            endpoint: Sanitized request method and URL, when available.
        """
        super().__init__(status, body, headers, message, endpoint)
        self.status = status
        self.body = body
        self.headers = headers
        self.endpoint = endpoint
        self._message = message

    @property
    def status_code(self) -> int:
        """Alias for `status`, matching common HTTP exception interfaces."""
        return self.status

    @property
    def request_id(self) -> str | None:
        """Return the TypeSafe request ID from the response headers, when present."""
        return self.headers.get(_REQUEST_ID_HEADER)

    @override
    def __str__(self) -> str:
        """Describe the failure without including its response body or headers."""
        try:
            reason = HTTPStatus(self.status).phrase
        except ValueError:
            reason = _SAFE_STATUS_MESSAGES.get(self.status, "API request failed")
        detail = self._message or reason
        message = f"{self.status} {detail}"
        if self.endpoint is not None:
            message = f"{self.endpoint}: {message}"
        if self.request_id is not None:
            message = f"{message} (request_id={self.request_id})"
        return message

    @override
    def __repr__(self) -> str:
        """Represent the error without including its response body or headers."""
        return f"{type(self).__name__}({str(self)!r})"


class TypeSafeBadRequestError(TypeSafeAPIError, ModelInvalidRequestError):
    """The TypeSafe request was invalid (HTTP 400)."""


class TypeSafeAuthenticationError(TypeSafeAPIError, ModelAuthenticationError):
    """Authentication with TypeSafe failed (HTTP 401)."""


class TypeSafePermissionDeniedError(TypeSafeAPIError, ModelPermissionDeniedError):
    """The TypeSafe credential cannot perform the request (HTTP 403)."""


class TypeSafeNotFoundError(TypeSafeAPIError, ModelNotFoundError):
    """The requested TypeSafe resource or model was not found (HTTP 404)."""


class TypeSafeUnprocessableEntityError(TypeSafeAPIError, ModelInvalidRequestError):
    """TypeSafe rejected the request body during validation (HTTP 422)."""


class TypeSafeRateLimitError(TypeSafeAPIError, ModelRateLimitError):
    """The TypeSafe rate limit was exceeded (HTTP 429).

    Attributes:
        retry_after_ms: Server-requested delay in milliseconds, or `None` when the
            response does not contain a valid retry header.
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx2.Headers,
        message: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Create a rate-limit error and parse its retry delay.

        Args:
            status: HTTP response status code.
            body: Parsed JSON body, plain response text, or `None`.
            headers: HTTP response headers.
            message: Optional safe message override.
            endpoint: Sanitized request method and URL, when available.
        """
        super().__init__(status, body, headers, message, endpoint)
        self.retry_after_ms = _parse_retry_after(headers)


class TypeSafeInternalServerError(TypeSafeAPIError, ModelAPIError):
    """TypeSafe failed to process the request (HTTP 5xx)."""


class TypeSafeAPIConnectionError(TypeSafeError, ModelConnectionError, ConnectionError):
    """A TypeSafe request failed without receiving an HTTP response."""


class TypeSafeAPITimeoutError(
    TypeSafeAPIConnectionError,
    ModelTimeoutError,
    TimeoutError,
):
    """A TypeSafe request exceeded its configured timeout.

    Attributes:
        timeout: Timeout setting used by the sync or async HTTP client.
    """

    def __init__(self, timeout: float | httpx2.Timeout) -> None:
        """Create a timeout error.

        Args:
            timeout: Timeout setting used for the failed request.
        """
        super().__init__(timeout)
        self.timeout = timeout

    @override
    def __str__(self) -> str:
        """Return the configured timeout without request or credential data."""
        return f"Request timed out (timeout={self.timeout})."

    @override
    def __repr__(self) -> str:
        """Represent the timeout using its safe formatted message."""
        return f"{type(self).__name__}({str(self)!r})"


class TypeSafeAPIResponseValidationError(TypeSafeAPIError):
    """A successful response was missing or contained invalid required data.

    Attributes:
        field_path: Dotted path to the first field that failed validation.
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx2.Headers,
        field_path: str,
        endpoint: str | None = None,
    ) -> None:
        """Create a response-validation error.

        Args:
            status: Successful HTTP response status code.
            body: Parsed JSON body or plain response text.
            headers: HTTP response headers.
            field_path: Dotted path to the first invalid field.
            endpoint: Sanitized request method and URL, when available.
        """
        self.field_path = field_path
        super().__init__(
            status,
            body,
            headers,
            f"Invalid response data at {field_path!r}.",
            endpoint,
        )
        self.args = (status, body, headers, field_path, endpoint)


def _parse_retry_after(headers: httpx2.Headers) -> float | None:
    for name, multiplier in (
        (_RETRY_AFTER_MS_HEADER, 1.0),
        (_RETRY_AFTER_HEADER, 1000.0),
    ):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip() or "0")
        except ValueError:
            if name == _RETRY_AFTER_HEADER:
                try:
                    delay = (
                        parsedate_to_datetime(raw).timestamp() - time.time()
                    ) * 1000
                except (OverflowError, TypeError, ValueError):
                    continue
                return max(0.0, delay)
            continue
        if math.isfinite(value) and value >= 0:
            delay = value * multiplier
            if math.isfinite(delay):
                return delay
        if name == _RETRY_AFTER_HEADER:
            return None
    return None


def _response_body(response: httpx2.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def _response_endpoint(response: httpx2.Response) -> str | None:
    try:
        request = response.request
    except RuntimeError:
        return None
    parts = urlsplit(str(request.url))
    hostname = parts.hostname
    if hostname is None:
        return None
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    url = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return f"{request.method} {url}"


_STATUS_ERROR_TYPES: dict[int, type[TypeSafeAPIError]] = {
    400: TypeSafeBadRequestError,
    401: TypeSafeAuthenticationError,
    403: TypeSafePermissionDeniedError,
    404: TypeSafeNotFoundError,
    422: TypeSafeUnprocessableEntityError,
    429: TypeSafeRateLimitError,
}


def _api_error(response: httpx2.Response) -> TypeSafeAPIError:
    error_type = _STATUS_ERROR_TYPES.get(
        response.status_code,
        TypeSafeInternalServerError
        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
        else TypeSafeAPIError,
    )
    return error_type(
        response.status_code,
        _response_body(response),
        response.headers,
        endpoint=_response_endpoint(response),
    )


def parse_response(response: httpx2.Response) -> ClassificationResponse:
    """Validate an HTTP response and convert it to a classification response.

    Args:
        response: Raw HTTP response returned by the TypeSafe API.

    Returns:
        Validated classification answers and metadata. The TypeSafe request ID is
        copied from the response headers when present.

    Raises:
        TypeSafeAPIError: If TypeSafe returns an unsuccessful status code. Specific
            statuses use subclasses that also inherit from LangChain model errors.
        TypeSafeAPIResponseValidationError: If a successful response is not valid JSON
            or does not match the expected response schema.
    """
    if not response.is_success:
        raise _api_error(response)
    endpoint = _response_endpoint(response)
    body = _response_body(response)
    try:
        parsed = ClassificationResponse.model_validate(body)
    except ValidationError as error:
        location = error.errors()[0].get("loc", ())
        field_path = ".".join(str(item) for item in location) or "response"
        raise TypeSafeAPIResponseValidationError(
            response.status_code,
            body,
            response.headers,
            field_path,
            endpoint,
        ) from error
    return parsed.model_copy(
        update={"request_id": response.headers.get(_REQUEST_ID_HEADER)}
    )


__all__ = [
    "TypeSafeAPIConnectionError",
    "TypeSafeAPIError",
    "TypeSafeAPIResponseValidationError",
    "TypeSafeAPITimeoutError",
    "TypeSafeAuthenticationError",
    "TypeSafeBadRequestError",
    "TypeSafeError",
    "TypeSafeInternalServerError",
    "TypeSafeNotFoundError",
    "TypeSafePermissionDeniedError",
    "TypeSafeRateLimitError",
    "TypeSafeUnprocessableEntityError",
    "parse_response",
]
