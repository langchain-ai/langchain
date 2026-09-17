"""LangChain runnable for TypeSafe classification."""

from __future__ import annotations

import logging
from typing import Any

import httpx2
from langchain_core._api import beta
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env, secret_from_env
from langsmith.run_helpers import get_current_run_tree
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self, override

from langchain_typesafe._state import serialize_state
from langchain_typesafe._version import __version__
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    parse_response,
)
from langchain_typesafe.types import ClassificationResponse, Question, State

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_MODEL = "jev-latest"
_DEFAULT_TIMEOUT = 30.0
_LS_PROVIDER = "typesafe"

logger = logging.getLogger(__name__)


@beta()
class TypeSafeClassifier(RunnableSerializable[State, ClassificationResponse]):
    """Classify JSON-compatible state with TypeSafe.

    `TypeSafeClassifier` is a LangChain `Runnable` for asking one or more typed
    questions about text or structured state. A single request can combine binary
    `Noul` judgments, categorical `Choice` classifications, and ordinal `Score`
    evaluations. The response preserves probabilities, confidence, and token usage so
    application code can decide whether to act, route, or request human review.

    After configuration is validated, the classifier creates both synchronous and
    asynchronous `httpx2` clients. Supply `client` and/or `async_client` to reuse
    clients configured by your application, including custom transports for testing,
    or network policy enforcement. Injected clients are used as-is; `timeout` only
    configures clients created by this class.

    The classifier does not add separate client lifecycle methods. Keep classifier
    instances long-lived to benefit from connection pooling. Applications that require
    deterministic cleanup can close `classifier.client` and `classifier.async_client`
    directly, following the corresponding `httpx2` sync and async client interfaces.

    Native TypeSafe state may be a string, JSON object, or JSON array. LangChain
    `BaseMessage` objects and message sequences can appear at the root or anywhere
    inside JSON objects and arrays. They are converted to role/content JSON before the
    request is sent. Message IDs are omitted, while system, user, assistant, and tool
    roles are preserved.

    The API key is read from `TYPESAFE_API_KEY` when `api_key` is omitted. Explicit
    constructor values take precedence over environment configuration.

    Args:
        questions: Named `Noul`, `Choice`, or `Score` questions. Names become keys in
            `ClassificationResponse.answers`.
        model: TypeSafe model used to answer the questions.
        api_key: TypeSafe API key. If omitted, reads `TYPESAFE_API_KEY`.
        base_url: Root URL for the TypeSafe API.
        timeout: Timeout, in seconds, applied to clients created by this class.
        client: Optional synchronous `httpx2.Client` used by `invoke`.
        async_client: Optional asynchronous `httpx2.AsyncClient` used by `ainvoke`.

    Raises:
        ValueError: If questions are empty, credentials are unavailable, or the timeout
            is not positive.

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

    ??? example "Classify asynchronously"

        `ainvoke` uses the classifier's asynchronous HTTP client and returns the same
        `ClassificationResponse` type as `invoke`.

        ```python
        from langchain_typesafe import Noul, TypeSafeClassifier

        classifier = TypeSafeClassifier(
            questions={
                "refund_requested": Noul(
                    instructions="Does the customer request a refund?"
                )
            }
        )

        response = await classifier.ainvoke("Please refund the duplicate charge.")
        print(response.nouls["refund_requested"].noul)
        ```
    """

    questions: dict[str, Question] = Field(min_length=1)
    """Questions sent together for every classifier invocation.

    The mapping key is the question ID and becomes the corresponding key in
    `ClassificationResponse.answers`. Question IDs identify answers for application
    code; put the complete judgment in each question's `instructions` rather than
    relying on its ID to provide model context.

    Questions share the same input state but are evaluated independently. Mix `Choice`,
    `Noul`, and `Score` questions in one mapping when several judgments use the same
    state instead of issuing one request per question.
    """

    model: str = Field(default=_DEFAULT_MODEL, min_length=1)
    """TypeSafe model name used for classification.

    The default, `jev-latest`, follows TypeSafe's latest compatible Jev release. Use a
    concrete model identifier when an application requires reproducible behavior across
    model updates. Leading and trailing whitespace is removed, and empty model names are
    rejected during initialization.
    """

    api_key: SecretStr | str = Field(
        default_factory=secret_from_env("TYPESAFE_API_KEY", default=""),
        exclude=True,
        repr=False,
    )
    """API key used to authenticate TypeSafe requests.

    If omitted, the key is read from the `TYPESAFE_API_KEY` environment variable when
    the classifier is initialized. An explicit constructor value takes precedence. The
    value is stored as `SecretStr` and excluded from model representation and
    serialization.

    ??? example "Specify with an environment variable"

        ```bash
        export TYPESAFE_API_KEY=...
        ```

        ```python
        from langchain_typesafe import Noul, TypeSafeClassifier

        classifier = TypeSafeClassifier(
            questions={"urgent": Noul(instructions="Is this urgent?")}
        )
        ```

    ??? example "Specify directly"

        ```python
        classifier = TypeSafeClassifier(
            api_key="...",
            questions={"urgent": Noul(instructions="Is this urgent?")},
        )
        ```
    """

    base_url: str = Field(
        default_factory=from_env("TYPESAFE_BASE_URL", default=_DEFAULT_BASE_URL)
    )
    """Root URL used for TypeSafe API requests.

    Resolution order:

    1. Explicit `base_url` supplied to `TypeSafeClassifier`.
    2. The `TYPESAFE_BASE_URL` environment variable.
    3. `https://api.typesafe.ai`.

    Requests are sent to `/v1/systemone` beneath this URL. Override it for a compatible
    gateway, test server, or private deployment. URL validation is delegated to
    `httpx2` when a request is made.
    """

    timeout: float = Field(default=_DEFAULT_TIMEOUT, gt=0)
    """Timeout in seconds for clients created by this classifier.

    This setting is passed to both `httpx2.Client` and `httpx2.AsyncClient` when their
    respective fields are omitted. It does not modify an injected client's timeout;
    configure custom clients directly when different sync and async policies are needed.
    """

    client: httpx2.Client | None = Field(default=None, exclude=True, repr=False)
    """Optional synchronous `httpx2.Client` used by `invoke` and `batch`.

    If omitted, the classifier creates a client using `timeout`. Supply a client to
    reuse connection pools or configure a custom transport, proxy, TLS policy, or test
    fixture. The injected client is used as-is and is not closed by the classifier; the
    caller retains responsibility for its lifecycle.

    This client is not used by `ainvoke` or `abatch`. Configure `async_client`
    separately when asynchronous calls also require custom HTTP behavior.
    """

    async_client: httpx2.AsyncClient | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    """Optional asynchronous `httpx2.AsyncClient` used by `ainvoke` and `abatch`.

    If omitted, the classifier creates an asynchronous client using `timeout`. Supply a
    client to reuse connection pools or configure a custom transport, proxy, TLS policy,
    or test fixture. The injected client is used as-is and is not closed by the
    classifier; the caller retains responsibility for its lifecycle.

    This client is not used by `invoke` or `batch`. Configure `client` separately when
    synchronous calls also require custom HTTP behavior.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str) -> str:
        model = model.strip()
        if not model:
            message = "TypeSafe model must not be empty."
            raise ValueError(message)
        return model

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, api_key: SecretStr | str) -> SecretStr:
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        if not secret.get_secret_value().strip():
            message = (
                "TypeSafe API key is required. Pass `api_key` or set "
                "`TYPESAFE_API_KEY`."
            )
            raise ValueError(message)
        return secret

    @model_validator(mode="after")
    def _build_clients(self) -> Self:
        """Create missing sync and async clients after configuration is validated."""
        if self.client is None:
            self.client = httpx2.Client(timeout=self.timeout)
        if self.async_client is None:
            self.async_client = httpx2.AsyncClient(timeout=self.timeout)
        return self

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
        return {"api_key": "TYPESAFE_API_KEY"}

    @override
    def invoke(
        self,
        input: State,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassificationResponse:
        """Classify one JSON-compatible input synchronously.

        Args:
            input: Text, object, array, `BaseMessage`, or message sequence to classify.
            config: Optional LangChain runnable configuration for callbacks, tags,
                metadata, and tracing.
            **_: Additional keyword arguments accepted for `Runnable` compatibility and
                otherwise ignored.

        Returns:
            Structured TypeSafe answers and request metadata.

        Raises:
            TypeError: If the input is not a supported state.
            TypeSafeAPIError: If TypeSafe returns an unsuccessful HTTP response.
            TypeSafeAPIConnectionError: If no HTTP response is received.
            TypeSafeAPITimeoutError: If the request exceeds its client timeout.
            TypeSafeAPIResponseValidationError: If a successful response is malformed.
        """
        return self._call_with_config(
            self._classify,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    @override
    async def ainvoke(
        self,
        input: State,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassificationResponse:
        """Classify one JSON-compatible input asynchronously.

        Args:
            input: Text, object, array, `BaseMessage`, or message sequence to classify.
            config: Optional LangChain runnable configuration for callbacks, tags,
                metadata, and tracing.
            **_: Additional keyword arguments accepted for `Runnable` compatibility and
                otherwise ignored.

        Returns:
            Structured TypeSafe answers and request metadata.

        Raises:
            TypeError: If the input is not a supported state.
            TypeSafeAPIError: If TypeSafe returns an unsuccessful HTTP response.
            TypeSafeAPIConnectionError: If no HTTP response is received.
            TypeSafeAPITimeoutError: If the request exceeds its client timeout.
            TypeSafeAPIResponseValidationError: If a successful response is malformed.
        """
        return await self._acall_with_config(
            self._aclassify,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    def _classify(self, state: State) -> ClassificationResponse:
        payload = self._payload(state)
        if self.client is None:  # pragma: no cover - guaranteed by model validation
            message = "Synchronous TypeSafe client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = self.client.post(
                self._endpoint,
                json=payload,
                headers=self._request_headers,
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the TypeSafe API."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    async def _aclassify(self, state: State) -> ClassificationResponse:
        payload = self._payload(state)
        if self.async_client is None:  # pragma: no cover - guaranteed by validation
            message = "Asynchronous TypeSafe client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = await self.async_client.post(
                self._endpoint,
                json=payload,
                headers=self._request_headers,
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the TypeSafe API."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """Set `ls_provider` and `ls_model_name` when run is created."""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": _LS_PROVIDER,
            "ls_model_name": self.model,
            "ls_model_type": "chat",
        }
        return config

    def _record_usage(self, response: ClassificationResponse) -> ClassificationResponse:
        """Attach TypeSafe token usage to the active run, if there is one.

        Nothing is written when tracing is disabled, and a tracing failure never fails
        an otherwise successful classification.
        """
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        try:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                run_tree.extra.setdefault("metadata", {})["usage_metadata"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
        except Exception:  # noqa: BLE001 - tracing must not break classification
            logger.debug("Could not attach TypeSafe usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    @property
    def _request_headers(self) -> dict[str, str]:
        api_key = (
            self.api_key.get_secret_value()
            if isinstance(self.api_key, SecretStr)
            else self.api_key
        )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"langchain-typesafe/{__version__}",
        }

    def _payload(self, state: State) -> dict[str, JsonValue]:
        return {
            "state": serialize_state(state),
            "model": self.model,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in self.questions.items()
            },
        }


__all__ = ["TypeSafeClassifier"]
