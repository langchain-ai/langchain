"""LangChain runnable for TypeSafe classification."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from typing import Any

import msgspec
import typesafe_sdk as ts
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self, override

from langchain_typesafe._errors import with_standard_errors
from langchain_typesafe._state import serialize_state
from langchain_typesafe._version import __version__
from langchain_typesafe.types import State

# The SDK sets its own `User-Agent` last and unconditionally, so integration
# attribution travels in a dedicated header instead.
_INTEGRATION_HEADER = "X-LangChain-Integration"
_INTEGRATION_VALUE = f"langchain-typesafe/{__version__}"

_QUESTION_TYPES: dict[str, type[ts.Noul | ts.Choice | ts.Score]] = {
    "noul": ts.Noul,
    "choice": ts.Choice,
    "score": ts.Score,
}


class TypeSafeClassifier(RunnableSerializable[State, ts.SystemOneResponse]):
    """Classify JSON-compatible state with TypeSafe.

    `TypeSafeClassifier` is a LangChain `Runnable` for asking one or more typed
    questions about text or structured state. A single request can combine binary
    `Noul` judgments, categorical `Choice` classifications, and ordinal `Score`
    evaluations. The response preserves probabilities, confidence, and token usage so
    application code can decide whether to act, route, or request human review.

    Requests are issued through the official TypeSafe Python SDK, so retries, backoff,
    error classification, and response validation follow the provider's own behavior.
    Questions and answers are the SDK's types; this class adds the LangChain
    `Runnable` interface, tracing through the supplied `RunnableConfig`, and support
    for LangChain messages inside the input state.

    Native TypeSafe state may be a string, JSON object, or JSON array. LangChain
    `BaseMessage` objects and message sequences can appear at the root or anywhere
    inside JSON objects and arrays. They are converted to role/content JSON before the
    request is sent. Message IDs are omitted, while system, user, assistant, and tool
    roles are preserved.

    Configuration that is left unset is resolved by the SDK, which reads
    `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, and `TYPESAFE_DEFAULT_MODEL` from the
    environment. Explicit constructor values take precedence.

    HTTP clients are created on first use and reused for the lifetime of the
    classifier, so keep instances long-lived to benefit from connection pooling. Call
    `close` or `aclose` when deterministic cleanup is required, or use the classifier
    as a context manager.

    Args:
        questions: Named `Noul`, `Choice`, or `Score` questions. Names become keys in
            `SystemOneResponse.answers`. Question dictionaries using a `type`
            discriminator are converted to the corresponding SDK type.
        model: TypeSafe model used to answer the questions. If omitted, the SDK
            resolves `TYPESAFE_DEFAULT_MODEL` or its own default.
        api_key: TypeSafe API key. If omitted, reads `TYPESAFE_API_KEY`.
        base_url: Root URL for the TypeSafe API. If omitted, reads
            `TYPESAFE_BASE_URL`.
        timeout: Request timeout in seconds. If omitted, the SDK default applies.
        retry: Retry policy for TypeSafe requests. If omitted, the SDK's default
            policy applies, which retries HTTP 408, 429, and 5xx responses as well as
            connection and timeout failures using exponential backoff that honors the
            provider's retry headers. Pass `RetryPolicy(max_retries=0)` to disable
            retries.
        client: Optional `typesafe_sdk.TypeSafeClient` used by `invoke` and `batch`.
        async_client: Optional `typesafe_sdk.AsyncTypeSafeClient` used by `ainvoke`
            and `abatch`.

    Raises:
        ValueError: If questions are empty, a question dictionary has an unknown
            `type`, credentials are unavailable, or the model name is blank.

    ??? example "Classify state on several dimensions"

        Send questions that share the same state together. Each answer remains
        independently addressable through its question ID.

        ```python
        from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

        classifier = TypeSafeClassifier(
            questions={
                "department": Choice(
                    instructions="Which team should handle this request?",
                    criteria={
                        "billing": "Payment or subscription issues.",
                        "technical": "Product bugs or integration failures.",
                    },
                ),
                "urgent": Noul(
                    instructions="Does this message require an urgent response?"
                ),
                "frustration": Score(
                    instructions="How frustrated does the customer appear?",
                    criteria=["Calm.", "Concerned but civil.", "Very angry."],
                ),
            }
        )

        response = classifier.invoke(
            "Stripe has failed to connect for three days. Please help immediately."
        )
        print(response.choices["department"].choice)
        print(response.nouls["urgent"].noul)
        print(response.scores["frustration"].score)
        ```

    ??? example "Classify agent messages asynchronously"

        `ainvoke` returns the same `SystemOneResponse` type as `invoke`. Passing
        messages directly is the most common way to reuse agent context.

        ```python
        from langchain_typesafe import Noul, TypeSafeClassifier

        classifier = TypeSafeClassifier(
            questions={
                "refund_requested": Noul(
                    instructions="Does the customer request a refund?"
                )
            }
        )

        async with classifier:
            response = await classifier.ainvoke(state.messages)

        print(response.nouls["refund_requested"].noul)
        ```

    ??? example "Handle failures with provider-independent errors"

        TypeSafe errors are both SDK exceptions and LangChain `ModelError` subclasses,
        so either import can be used to handle them.

        ```python
        from langchain_core.exceptions import ModelRateLimitError

        try:
            response = classifier.invoke("...")
        except ModelRateLimitError as error:
            print(error.retry_after_ms, error.request_id)
        ```
    """

    questions: Mapping[str, ts.Noul | ts.Choice | ts.Score] = Field(min_length=1)
    """Questions sent together for every classifier invocation.

    The mapping key is the question ID and becomes the corresponding key in
    `SystemOneResponse.answers`. Question IDs identify answers for application code and
    are not sent to the model; put the complete judgment in each question's
    `instructions` rather than relying on its ID to provide context.

    Questions share the same input state but are evaluated independently. Mix `Choice`,
    `Noul`, and `Score` questions in one mapping when several judgments use the same
    state instead of issuing one request per question.
    """

    model: str | None = None
    """TypeSafe model name used for classification.

    When omitted, the SDK resolves the `TYPESAFE_DEFAULT_MODEL` environment variable
    and otherwise falls back to its own default. Set a concrete model identifier when
    an application requires reproducible behavior across model updates. Leading and
    trailing whitespace is removed, and blank model names are rejected.
    """

    api_key: SecretStr | str | None = Field(default=None, exclude=True, repr=False)
    """API key used to authenticate TypeSafe requests.

    If omitted, the key is read from the `TYPESAFE_API_KEY` environment variable. An
    explicit constructor value takes precedence. The value is stored as `SecretStr`,
    excluded from model representation and serialization, and unwrapped only when a
    TypeSafe client is constructed.
    """

    base_url: str | None = None
    """Root URL used for TypeSafe API requests.

    When omitted, the SDK resolves the `TYPESAFE_BASE_URL` environment variable and
    otherwise uses the public API. Override it for a compatible gateway, test server,
    or private deployment.
    """

    timeout: float | None = None
    """Request timeout in seconds for clients created by this classifier.

    This does not modify an injected client's timeout; configure custom clients
    directly when different sync and async policies are needed.
    """

    retry: ts.RetryPolicy | None = Field(default=None, exclude=True)
    """Retry policy applied to clients created by this classifier.

    When omitted, the SDK's default policy applies. It retries HTTP 408, 429, and 5xx
    responses along with connection and timeout failures, using exponential backoff
    with jitter that honors the `Retry-After` and `retry-after-ms` response headers.
    Because TypeSafe asks clients to back off and retry on 429 and 529, leaving this
    unset is recommended. Pass `RetryPolicy(max_retries=0)` to disable retries.
    """

    client: ts.TypeSafeClient | None = Field(default=None, exclude=True, repr=False)
    """Optional synchronous TypeSafe client used by `invoke` and `batch`.

    If omitted, the classifier creates one on first use from the configuration on this
    class. Supply a client to reuse connection pools or to configure a custom
    transport, proxy, or test fixture. An injected client is used as-is and is not
    closed by `close`; the caller retains responsibility for its lifecycle.
    """

    async_client: ts.AsyncTypeSafeClient | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    """Optional asynchronous TypeSafe client used by `ainvoke` and `abatch`.

    If omitted, the classifier creates one on first use from the configuration on this
    class. Supply a client to reuse connection pools or to configure a custom
    transport, proxy, or test fixture. An injected client is used as-is and is not
    closed by `aclose`; the caller retains responsibility for its lifecycle.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    _owns_client: bool = PrivateAttr(default=False)
    _owns_async_client: bool = PrivateAttr(default=False)
    _client_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, questions: Any) -> Any:
        """Convert question dictionaries to their corresponding SDK question types."""
        if not isinstance(questions, dict):
            return questions
        coerced: dict[Any, Any] = {}
        for name, question in questions.items():
            if not isinstance(question, dict):
                coerced[name] = question
                continue
            fields = dict(question)
            kind = fields.pop("type", None)
            question_type = _QUESTION_TYPES.get(kind) if isinstance(kind, str) else None
            if question_type is None:
                msg = (
                    f"Question {name!r} has unsupported type {kind!r}. Expected one of "
                    f"{sorted(_QUESTION_TYPES)}."
                )
                raise ValueError(msg)
            coerced[name] = question_type(**fields)
        return coerced

    @field_serializer("questions")
    def _serialize_questions(
        self,
        questions: Mapping[str, ts.Noul | ts.Choice | ts.Score],
    ) -> dict[str, Any]:
        """Serialize SDK question types for LangChain serialization and tracing.

        The SDK models questions as `msgspec` structs, which pydantic cannot serialize
        on its own. The resulting dictionaries carry the `type` discriminator, so a
        serialized classifier round-trips through `_coerce_questions`.
        """
        return {name: msgspec.to_builtins(q) for name, q in questions.items()}

    @field_validator("api_key")
    @classmethod
    def _coerce_api_key(cls, api_key: SecretStr | str | None) -> SecretStr | None:
        """Wrap a plain string key so it is never exposed by accident."""
        if api_key is None or isinstance(api_key, SecretStr):
            return api_key
        return SecretStr(api_key)

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str | None) -> str | None:
        if model is None:
            return None
        model = model.strip()
        if not model:
            msg = "TypeSafe model must not be empty."
            raise ValueError(msg)
        return model

    @model_validator(mode="after")
    def _validate_credentials(self) -> Self:
        """Fail at construction time when no API key can be resolved.

        Clients are built lazily, so without this check a missing key would only
        surface on the first invocation.
        """
        if self.client is not None and self.async_client is not None:
            return self
        if (
            isinstance(self.api_key, SecretStr)
            and self.api_key.get_secret_value().strip()
        ):
            return self
        if os.environ.get(ts.constants.API_KEY_ENV, "").strip():
            return self
        msg = (
            "TypeSafe API key is required. Pass `api_key` or set "
            f"`{ts.constants.API_KEY_ENV}`."
        )
        raise ValueError(msg)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "classifiers", "typesafe"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map the API-key field to its environment variable for serialization."""
        return {"api_key": ts.constants.API_KEY_ENV}

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "api_key": (
                self.api_key.get_secret_value()
                if isinstance(self.api_key, SecretStr)
                else None
            ),
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "retry": self.retry,
            "headers": {_INTEGRATION_HEADER: _INTEGRATION_VALUE},
        }

    def _sync_client(self) -> ts.TypeSafeClient:
        if self.client is None:
            with self._client_lock:
                if self.client is None:
                    self.client = ts.TypeSafeClient(**self._client_kwargs())
                    self._owns_client = True
        return self.client

    def _get_async_client(self) -> ts.AsyncTypeSafeClient:
        if self.async_client is None:
            with self._client_lock:
                if self.async_client is None:
                    self.async_client = ts.AsyncTypeSafeClient(**self._client_kwargs())
                    self._owns_async_client = True
        return self.async_client

    @override
    def invoke(
        self,
        input: State,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ts.SystemOneResponse:
        """Classify one JSON-compatible input synchronously.

        Args:
            input: Text, object, array, `BaseMessage`, or message sequence to classify.
            config: Optional LangChain runnable configuration for callbacks, tags,
                metadata, and tracing.
            **kwargs: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            The TypeSafe `SystemOneResponse` for this request, carrying answers keyed
            by question ID along with the model, token usage, and request ID.

        Raises:
            TypeError: If the input is not a supported state.
            TypeSafeAPIError: If TypeSafe returns an unsuccessful HTTP response.
                Classified statuses raise subclasses that are also
                `langchain_core.exceptions.ModelError` subclasses.
            TypeSafeAPIConnectionError: If no HTTP response is received.
            TypeSafeAPITimeoutError: If the request exceeds its timeout.
            TypeSafeAPIResponseValidationError: If a successful response is malformed.
        """
        return self._call_with_config(self._classify, input, config, run_type="chain")

    @override
    async def ainvoke(
        self,
        input: State,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ts.SystemOneResponse:
        """Classify one JSON-compatible input asynchronously.

        Args:
            input: Text, object, array, `BaseMessage`, or message sequence to classify.
            config: Optional LangChain runnable configuration for callbacks, tags,
                metadata, and tracing.
            **kwargs: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            The TypeSafe `SystemOneResponse` for this request, carrying answers keyed
            by question ID along with the model, token usage, and request ID.

        Raises:
            TypeError: If the input is not a supported state.
            TypeSafeAPIError: If TypeSafe returns an unsuccessful HTTP response.
                Classified statuses raise subclasses that are also
                `langchain_core.exceptions.ModelError` subclasses.
            TypeSafeAPIConnectionError: If no HTTP response is received.
            TypeSafeAPITimeoutError: If the request exceeds its timeout.
            TypeSafeAPIResponseValidationError: If a successful response is malformed.
        """
        return await self._acall_with_config(
            self._aclassify,
            input,
            config,
            run_type="chain",
        )

    def _classify(self, state: State) -> ts.SystemOneResponse:
        payload = serialize_state(state)
        client = self._sync_client()
        with with_standard_errors():
            return client.system_one(payload, self.questions)

    async def _aclassify(self, state: State) -> ts.SystemOneResponse:
        payload = serialize_state(state)
        client = self._get_async_client()
        with with_standard_errors():
            return await client.system_one(payload, self.questions)

    def close(self) -> None:
        """Close the synchronous client created by this classifier.

        Injected clients are left open because the caller owns their lifecycle.
        """
        if self.client is not None and self._owns_client:
            self.client.close()
            self.client = None
            self._owns_client = False

    async def aclose(self) -> None:
        """Close the asynchronous client created by this classifier.

        Injected clients are left open because the caller owns their lifecycle.
        """
        if self.async_client is not None and self._owns_async_client:
            await self.async_client.aclose()
            self.async_client = None
            self._owns_async_client = False

    def __enter__(self) -> Self:
        """Return the classifier for use inside a `with` block."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the synchronous client created by this classifier."""
        self.close()

    async def __aenter__(self) -> Self:
        """Return the classifier for use inside an `async with` block."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the asynchronous client created by this classifier."""
        await self.aclose()


__all__ = ["TypeSafeClassifier"]
