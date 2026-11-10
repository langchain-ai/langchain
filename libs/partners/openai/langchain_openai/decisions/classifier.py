"""Decision model for the OpenAI Decisions API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import openai
from langchain_core._api import beta
from langchain_core.decisions import BaseDecisionModel, ClassifierResponse
from langchain_core.exceptions import ModelRefusalError
from langchain_core.utils import from_env
from langchain_core.utils._gateway import _resolve_gateway_config
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
from langchain_openai.decisions._classifier_input import (
    _to_classifier_input,
    _to_classifier_questions,
)

if TYPE_CHECKING:
    from langchain_core.decisions import Question, State


@beta()
class OpenAIClassifier(BaseDecisionModel):
    """Ask typed questions about text, messages, or images with OpenAI Decisions.

    A single request can combine `Noul` probabilities, `Choice` selections, and
    `Score` ratings over shared state. The response preserves probabilities,
    confidence, and token usage so application code can decide whether to act,
    route, or request human review.

    Messages are sent as native Decisions API input items, so tool calls and tool
    results keep their roles. Other structured state is sent as JSON.

    !!! warning "Unsupported features"

        - `Box` questions raise `ValueError`.
        - A `Choice` needs at least two labels.
        - The API's `predicate` question has no outcome descriptions, so `Noul`
            criteria are appended to the instructions as text.
        - The API answers each question or refuses it. If any question is refused,
            `invoke` raises `ModelRefusalError` and the other answers are discarded.
        - Prompt caching options, `safety_identifier`, and cache breakpoints are
            not exposed.
        - Only text and image content is accepted. Images may be data URLs or
            public `http(s)` URLs; files and audio raise `ValueError`.
        - Structured state, such as a dict, is sent as JSON, which cannot carry
            media. Media blocks in it, including in nested messages, are dropped
            with a warning. Pass messages as the root state to include their images.

    Args:
        model: Decisions model used to answer questions.
        api_key: OpenAI API key. If omitted, reads `OPENAI_API_KEY`, or the LangSmith
            gateway key when routing through the gateway.
        base_url: Base URL for API requests. If omitted, reads `OPENAI_API_BASE`, then
            the LangSmith gateway when `LANGSMITH_GATEWAY` is set, then
            `OPENAI_BASE_URL`.
        organization: OpenAI organization ID. If omitted, reads `OPENAI_ORG_ID`.
        timeout: Request timeout passed to the OpenAI client.
        max_retries: Maximum number of retries passed to the OpenAI client.
        default_headers: Headers sent with every request.
        default_query: Query parameters sent with every request.
        http_client: Optional `httpx.Client` used by sync invocations.
        http_async_client: Optional `httpx.AsyncClient` used by async invocations.

    Raises:
        ValueError: If no API key is available. Also raised by `invoke` for a
            question or state the API does not support.
        ModelRefusalError: From `invoke` when the API refused any question.

    ??? example "Ask several questions about the same state"

        ```python
        from langchain_core.decisions import Choice, Noul, Score
        from langchain_openai.decisions import OpenAIClassifier

        classifier = OpenAIClassifier(model="gpt-6-luna")
        response = classifier.invoke(
            {
                "state": "Stripe has failed to connect for three days. Help ASAP.",
                "questions": {
                    "department": Choice(
                        instructions="Which team should handle this request?",
                        criteria={
                            "billing": "Payment or subscription issues.",
                            "technical": "Product bugs or integration failures.",
                        },
                    ),
                    "urgent": Noul(instructions="Is this request urgent?"),
                    "frustration": Score(
                        instructions="How frustrated does the customer appear?",
                        criteria=["Calm.", "Concerned but civil.", "Very angry."],
                    ),
                },
            }
        )
        print(response.choices["department"].choice)
        print(response.nouls["urgent"].noul)
        print(response.scores["frustration"].score)
        ```
    """

    model: str = Field(min_length=1)
    """Decisions model name, such as `gpt-6-luna`."""

    api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] = None
    """API key used to authenticate requests.

    Automatically inferred from env var `OPENAI_API_KEY` if not provided. When
    requests are routed through the LangSmith gateway, `LANGSMITH_GATEWAY_API_KEY`
    or `LANGSMITH_API_KEY` is preferred instead.
    """

    base_url: str | None = None
    """Base URL for API requests.

    Resolution order (first match wins):

    1. Explicit `base_url` kwarg.
    2. Env var `OPENAI_API_BASE` (read by LangChain at init).
    3. The [LangSmith LLM gateway](https://docs.langchain.com/langsmith/llm-gateway),
        when env var `LANGSMITH_GATEWAY` is set.
    4. Env var `OPENAI_BASE_URL` (read by the underlying `openai` SDK client).
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
    def _validate_environment(self) -> Self:
        gateway_config = _resolve_gateway_config(
            base_url=self.base_url,
            api_key=self.api_key,
            provider_path="openai/v1",
            base_url_env="OPENAI_API_BASE",
            api_key_env="OPENAI_API_KEY",
        )
        self.base_url = gateway_config.base_url
        self.api_key = gateway_config.api_key
        if self.api_key is None or (
            isinstance(self.api_key, SecretStr)
            and not self.api_key.get_secret_value().strip()
        ):
            msg = (
                "OpenAI API key is required. Pass `api_key`, set `OPENAI_API_KEY`, "
                "or set `LANGSMITH_GATEWAY` with a LangSmith API key."
            )
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

    @property
    @override
    def _provider(self) -> str:
        return "openai"

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
    def _classify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        payload = self._payload(state, questions)
        try:
            response = self._client.post(
                "/decisions", body=payload, cast_to=httpx.Response
            )
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        return _parse_response(response, questions)

    @override
    async def _aclassify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        payload = self._payload(state, questions)
        try:
            response = await self._async_client.post(
                "/decisions", body=payload, cast_to=httpx.Response
            )
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        return _parse_response(response, questions)

    def _payload(self, state: State, questions: dict[str, Question]) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": _to_classifier_input(state),
            "questions": _to_classifier_questions(questions),
        }


def _parse_response(
    response: httpx.Response, questions: dict[str, Question]
) -> ClassifierResponse:
    try:
        body = response.json()
        answers = body["answers"]
        if len(answers) != len(questions):
            msg = "Expected one answer per question."
            raise ValueError(msg)
        pairs = list(zip(questions.items(), answers, strict=True))
        refused = [name for (name, _), answer in pairs if answer["type"] == "refusal"]
        if refused:
            msg = f"The OpenAI Decisions API refused to answer: {', '.join(refused)}."
            raise ModelRefusalError(msg)
        return ClassifierResponse.model_validate(
            {
                "model": body["model"],
                "answers": {
                    name: _to_answer(question, answer)
                    for (name, question), answer in pairs
                },
                "usage": _to_usage(body.get("usage")),
                "request_id": response.headers.get("x-request-id"),
            }
        )
    except (ValueError, KeyError, TypeError, ValidationError) as e:
        msg = "Invalid response from the OpenAI Decisions API."
        raise openai.APIResponseValidationError(
            response=cast("Any", response), body=None, message=msg
        ) from e


def _to_answer(question: Question, answer: dict[str, Any]) -> dict[str, Any]:
    if answer["type"] == "predicate":
        return {"type": "noul", "noul": answer["probability"]}
    probabilities = {
        item["value"]: item["probability"] for item in answer["probabilities"]
    }
    if answer["type"] == "choice":
        return {
            "type": "choice",
            "choice": answer["choice"],
            "probabilities": probabilities,
            "confidence": answer["confidence"],
        }
    return {
        "type": "score",
        "score": answer["score"],
        "legend": dict(enumerate(question.criteria)),  # type: ignore[union-attr, arg-type]
        "probabilities": probabilities,
        "confidence": answer["confidence"],
    }


def _to_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not usage:
        return None
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "input_token_details": {
            "cache_read": input_details.get("cached_tokens", 0),
            "cache_creation": input_details.get("cache_write_tokens", 0),
        },
        "output_token_details": {
            "reasoning": output_details.get("reasoning_tokens", 0),
        },
    }


__all__ = ["OpenAIClassifier"]
