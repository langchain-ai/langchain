"""LangChain runnable for the OpenAI Decisions API."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from functools import cached_property
from typing import Any, cast

import openai
from langchain_core._api import beta
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env, secret_from_env
from langsmith.run_helpers import get_current_run_tree
from pydantic import ConfigDict, Field, SecretStr, ValidationError, model_validator
from typing_extensions import Self, override

from langchain_openai._compat import httpx
from langchain_openai.chat_models._client_utils import (
    _resolve_sync_and_async_api_keys,
)
from langchain_openai.chat_models.base import (
    _handle_openai_api_error,
    _handle_openai_bad_request,
)
from langchain_openai.decisions._input import to_decision_input
from langchain_openai.decisions.types import DecisionRequest, DecisionResponse

_LS_PROVIDER = "openai"
_ANSWER_TYPES = frozenset({"predicate", "choice", "score", "refusal"})

logger = logging.getLogger(__name__)


@beta()
class OpenAIDecisions(RunnableSerializable[DecisionRequest, DecisionResponse]):
    """Ask typed questions about text or images with the OpenAI Decisions API.

    A single request can combine `Predicate` probabilities, `Choice` selections, and
    `Score` ratings over shared input. The response preserves probabilities,
    confidence, and token usage so application code can decide whether to act,
    route, or request human review.

    Sync and async OpenAI clients are created on first use of `invoke` or `ainvoke`
    respectively.

    Args:
        model: Decisions model used to answer questions.
        api_key: OpenAI API key. If omitted, reads `OPENAI_API_KEY`.
        base_url: Base URL for API requests. If omitted, reads `OPENAI_BASE_URL`.
        organization: OpenAI organization ID. If omitted, reads `OPENAI_ORG_ID`.
        timeout: Request timeout passed to the OpenAI client.
        max_retries: Maximum number of retries passed to the OpenAI client.
        default_headers: Headers sent with every request.
        default_query: Query parameters sent with every request.
        http_client: Optional `httpx.Client` used by sync invocations.
        http_async_client: Optional `httpx.AsyncClient` used by async invocations.

    Raises:
        ValueError: If no API key is available.

    ??? example "Ask several questions about the same input"

        ```python
        from langchain_openai.decisions import (
            Choice,
            OpenAIDecisions,
            Predicate,
            Score,
        )

        decisions = OpenAIDecisions(model="gpt-6-luna")
        response = decisions.invoke(
            {
                "input": "Stripe has failed to connect for three days. Help ASAP.",
                "questions": {
                    "department": Choice(
                        instructions="Which team should handle this request?",
                        choices={
                            "billing": "Payment or subscription issues.",
                            "technical": "Product bugs or integration failures.",
                        },
                    ),
                    "urgent": Predicate(instructions="Is this request urgent?"),
                    "frustration": Score(
                        instructions="How frustrated does the customer appear?",
                        levels=["Calm", "Concerned", "Angry"],
                    ),
                },
            }
        )
        print(response.choices["department"].choice)
        print(response.predicates["urgent"].probability)
        print(response.scores["frustration"].score)
        ```
    """

    model: str = Field(min_length=1)
    """Decisions model name, such as `gpt-6-luna`."""

    api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] = (
        Field(
            default_factory=secret_from_env("OPENAI_API_KEY", default=None),
            exclude=True,
            repr=False,
        )
    )
    """API key used to authenticate requests.

    Automatically inferred from env var `OPENAI_API_KEY` if not provided.
    """

    base_url: str | None = Field(
        default_factory=from_env("OPENAI_BASE_URL", default=None)
    )
    """Base URL for API requests.

    Automatically inferred from env var `OPENAI_BASE_URL` if not provided. When unset,
    requests go to the default OpenAI endpoint.
    """

    openai_organization: str | None = Field(
        alias="organization",
        default_factory=from_env(
            ["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None
        ),
    )
    """OpenAI organization ID.

    Automatically inferred from env var `OPENAI_ORG_ID` if not provided.
    """

    timeout: float | tuple[float, float] | Any | None = None
    """Request timeout. Can be float, `httpx.Timeout`, or `None`."""

    max_retries: int | None = None
    """Maximum number of retries. Uses the OpenAI SDK default when `None`."""

    default_headers: Mapping[str, str] | None = None
    """Headers sent with every request."""

    default_query: Mapping[str, object] | None = None
    """Query parameters sent with every request."""

    http_client: Any | None = Field(default=None, exclude=True, repr=False)
    """Optional `httpx.Client` used by `invoke` and `batch`.

    Injected clients are used as-is; the caller retains responsibility for their
    lifecycle.
    """

    http_async_client: Any | None = Field(default=None, exclude=True, repr=False)
    """Optional `httpx.AsyncClient` used by `ainvoke` and `abatch`.

    Injected clients are used as-is; the caller retains responsibility for their
    lifecycle.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def _validate_api_key(self) -> Self:
        if self.api_key is None or (
            isinstance(self.api_key, SecretStr)
            and not self.api_key.get_secret_value().strip()
        ):
            msg = "OpenAI API key is required. Pass `api_key` or set `OPENAI_API_KEY`."
            raise ValueError(msg)
        return self

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_openai", "decisions"]

    @property
    @override
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "OPENAI_API_KEY"}

    @cached_property
    def _client(self) -> openai.OpenAI:
        sync_api_key, _ = _resolve_sync_and_async_api_keys(self.api_key)  # type: ignore[arg-type]
        if sync_api_key is None:
            msg = (
                "Sync invocation requires a string or sync callable `api_key`. "
                "Use `ainvoke` with an async callable `api_key`."
            )
            raise ValueError(msg)
        return openai.OpenAI(
            api_key=sync_api_key,
            http_client=self.http_client,
            **self._client_params,
        )

    @cached_property
    def _async_client(self) -> openai.AsyncOpenAI:
        _, async_api_key = _resolve_sync_and_async_api_keys(self.api_key)  # type: ignore[arg-type]
        return openai.AsyncOpenAI(
            api_key=async_api_key,
            http_client=self.http_async_client,
            **self._client_params,
        )

    @property
    def _client_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "organization": self.openai_organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            params["max_retries"] = self.max_retries
        return params

    @override
    def invoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Answer one request synchronously.

        Args:
            input: Input and questions to evaluate.
            config: Optional runnable configuration for callbacks, tags, metadata,
                and tracing.
            **_: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            Typed answers keyed by question name, with request metadata.
        """
        return self._call_with_config(
            self._decide,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    @override
    async def ainvoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Answer one request asynchronously.

        Args:
            input: Input and questions to evaluate.
            config: Optional runnable configuration for callbacks, tags, metadata,
                and tracing.
            **_: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            Typed answers keyed by question name, with request metadata.
        """
        return await self._acall_with_config(
            self._adecide,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        payload = self._payload(request)
        try:
            response = self._client.post(
                "/decisions", body=payload, cast_to=httpx.Response
            )
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        return self._record_usage(_parse_response(response))

    async def _adecide(self, request: DecisionRequest) -> DecisionResponse:
        payload = self._payload(request)
        try:
            response = await self._async_client.post(
                "/decisions", body=payload, cast_to=httpx.Response
            )
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        return self._record_usage(_parse_response(response))

    def _payload(self, request: DecisionRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": to_decision_input(request["input"]),
            "questions": [
                question._to_api(name)  # noqa: SLF001
                for name, question in request["questions"].items()
            ],
        }

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

    def _record_usage(self, response: DecisionResponse) -> DecisionResponse:
        """Attach token usage to the active run, if there is one.

        Nothing is written when tracing is disabled, and a tracing failure never fails
        an otherwise successful request.
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
        except Exception:  # tracing must not break decisions
            logger.debug("Could not attach OpenAI Decisions usage.", exc_info=True)
        return response


def _parse_response(response: httpx.Response) -> DecisionResponse:
    try:
        body = response.json()
        return DecisionResponse.model_validate(
            {
                "model": body["model"],
                "answers": {
                    answer["name"]: _normalize_answer(answer)
                    for answer in body["answers"]
                    if answer.get("type") in _ANSWER_TYPES
                },
                "usage": body.get("usage") or {},
                "request_id": response.headers.get("x-request-id"),
            }
        )
    except (ValueError, KeyError, TypeError, ValidationError) as e:
        msg = "Invalid response from the OpenAI Decisions API."
        raise openai.APIResponseValidationError(
            response=cast("Any", response), body=None, message=msg
        ) from e


def _normalize_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Convert list-shaped probabilities to mappings keyed by value or index."""
    answer = {key: value for key, value in answer.items() if key != "name"}
    if answer["type"] == "choice":
        answer["probabilities"] = {
            item["value"]: item["probability"] for item in answer["probabilities"]
        }
    elif answer["type"] == "score":
        levels = answer["probabilities"]
        answer["legend"] = {item["value"]: item["label"] for item in levels}
        answer["probabilities"] = {
            item["value"]: item["probability"] for item in levels
        }
    return answer


__all__ = ["OpenAIDecisions"]
