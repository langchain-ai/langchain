"""Shared TypeSafe client plumbing for the middleware in this package.

Every middleware asks TypeSafe a fixed set of questions about some slice of agent
state. This module owns the small amount of shared work that involves: building the
sync and async SDK clients, issuing the request, translating errors, and reporting
failures without copying provider response bodies into application logs.

This is deliberately not a `Runnable`. Middleware run at fixed hook points rather than
being composed into chains, so wrapping one SDK call in a runnable would add an
indirection with nothing to compose.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import typesafe_sdk as ts
from langchain_core.utils import secret_from_env
from pydantic import SecretStr

from langchain_typesafe._errors import with_standard_errors
from langchain_typesafe._version import __version__

# The SDK sets its own `User-Agent` last and unconditionally, so integration
# attribution travels in a dedicated header instead.
_INTEGRATION_HEADER = "X-LangChain-Integration"
_INTEGRATION_VALUE = f"langchain-typesafe/{__version__}"

Questions = Mapping[str, ts.Noul | ts.Choice | ts.Score]
"""Questions asked on every request, keyed by question ID."""


class TypeSafeClassifier:
    """Issue one fixed TypeSafe request on behalf of a middleware.

    Args:
        questions: Questions asked on every request, keyed by question ID.
        api_key: TypeSafe API key. If omitted, reads `TYPESAFE_API_KEY`.
        base_url: Root URL for the TypeSafe API. If omitted, the SDK resolves
            `TYPESAFE_BASE_URL`.
        model: TypeSafe model used to answer the questions. If omitted, the SDK
            resolves `TYPESAFE_DEFAULT_MODEL` or its own default.
        timeout: Request timeout in seconds. If omitted, the SDK default applies.
        retry: Retry policy. If omitted, the SDK's default policy applies.
        client: Optional synchronous client, used as-is instead of creating one.
        async_client: Optional asynchronous client, used as-is.

    Raises:
        TypeSafeError: If the SDK cannot resolve an API key.
    """

    def __init__(
        self,
        questions: Questions,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        retry: ts.RetryPolicy | None = None,
        client: ts.TypeSafeClient | None = None,
        async_client: ts.AsyncTypeSafeClient | None = None,
    ) -> None:
        """Build the clients used for every classification."""
        if not questions:
            msg = "At least one question is required."
            raise ValueError(msg)
        self._questions = dict(questions)
        resolved_key = (
            api_key
            if api_key is not None
            else secret_from_env(ts.constants.API_KEY_ENV, default=None)()
        )
        secret = (
            SecretStr(resolved_key) if isinstance(resolved_key, str) else resolved_key
        )
        kwargs = {
            "api_key": secret.get_secret_value() if secret is not None else None,
            "base_url": base_url,
            "model": model,
            "timeout": timeout,
            "retry": retry,
            "headers": {_INTEGRATION_HEADER: _INTEGRATION_VALUE},
        }
        self._client = client if client is not None else ts.TypeSafeClient(**kwargs)  # type: ignore[arg-type]
        self._async_client = (
            async_client
            if async_client is not None
            else ts.AsyncTypeSafeClient(**kwargs)  # type: ignore[arg-type]
        )

    def classify(self, state: ts.JSONContent) -> ts.SystemOneResponse:
        """Classify state synchronously.

        Args:
            state: Text or JSON describing what should be classified. Middleware
                convert any LangChain messages before calling.

        Returns:
            Answers keyed by the question IDs supplied to this classifier.

        Raises:
            TypeSafeAPIError: If TypeSafe returns an unsuccessful HTTP response.
                Classified statuses raise subclasses that are also
                `langchain_core.exceptions.ModelError` subclasses.
        """
        with with_standard_errors():
            return self._client.system_one(state, self._questions)

    async def aclassify(self, state: ts.JSONContent) -> ts.SystemOneResponse:
        """Classify state asynchronously.

        Args:
            state: Text or JSON describing what should be classified. Middleware
                convert any LangChain messages before calling.

        Returns:
            Answers keyed by the question IDs supplied to this classifier.

        Raises:
            TypeSafeAPIError: If TypeSafe returns an unsuccessful HTTP response.
                Classified statuses raise subclasses that are also
                `langchain_core.exceptions.ModelError` subclasses.
        """
        with with_standard_errors():
            return await self._async_client.system_one(state, self._questions)


def log_classification_failure(
    logger: logging.Logger,
    error: Exception,
    fallback: str,
) -> None:
    """Record a classification failure without copying provider content into logs.

    The SDK embeds a truncated response body in `str(error)`, and a validation error
    can echo back part of the classified state. Agent state routinely contains user
    content, so only the error type and provider request metadata are logged.

    Args:
        logger: Logger belonging to the middleware that failed.
        error: Exception raised while classifying.
        fallback: Description of the behavior being used instead.
    """
    logger.warning(
        "TypeSafe classification failed (%s, status=%s, request_id=%s); %s",
        type(error).__name__,
        getattr(error, "status", None),
        getattr(error, "request_id", None),
        fallback,
    )


__all__ = ["Questions", "TypeSafeClassifier", "log_classification_failure"]
