"""LangChain runnable for the OpenAI Decisions API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from functools import cached_property
from typing import Any, cast

import openai
from langchain_core._api import beta
from langchain_core.decisions import (
    BaseDecisionModel,
    Choice,
    DecisionRequest,
    DecisionResponse,
    Question,
    Score,
)
from langchain_core.utils import from_env, secret_from_env
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

_ANSWER_TYPES = frozenset({"predicate", "choice", "score", "refusal"})


@beta()
class OpenAIDecisions(BaseDecisionModel):
    """Ask typed questions about text or images with the OpenAI Decisions API.

    A single request can combine `Predicate` probabilities, `Choice` selections, and
    `Score` ratings over shared input. The response preserves probabilities,
    confidence, and token usage so application code can decide whether to act,
    route, or request human review.

    The API natively accepts a string or user messages with text and base64 images.
    Strings and `HumanMessage` objects (alone or in a sequence) are sent natively;
    any other input, such as conversations with system or AI messages or JSON
    objects, is serialized to JSON text first.

    Sync and async OpenAI clients are created on first use of `invoke` or `ainvoke`
    respectively.

    Args:
        model: Decisions model used to answer questions.
        api_key: OpenAI API key. If omitted, reads `OPENAI_API_KEY`.
        base_url: Base URL for API requests. If omitted, reads `OPENAI_API_BASE`.
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

    openai_api_key: (
        SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]]
    ) = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENAI_API_KEY", default=None),
        exclude=True,
        repr=False,
    )
    """API key used to authenticate requests.

    Automatically inferred from env var `OPENAI_API_KEY` if not provided.
    """

    openai_api_base: str | None = Field(
        alias="base_url", default_factory=from_env("OPENAI_API_BASE", default=None)
    )
    """Base URL for API requests.

    Automatically inferred from env var `OPENAI_API_BASE` if not provided. When unset,
    the OpenAI SDK falls back to `OPENAI_BASE_URL`, then the default OpenAI endpoint.
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

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
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
        if self.openai_api_key is None or (
            isinstance(self.openai_api_key, SecretStr)
            and not self.openai_api_key.get_secret_value().strip()
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
        return {"openai_api_key": "OPENAI_API_KEY"}

    @cached_property
    def _client(self) -> openai.OpenAI:
        sync_api_key, _ = _resolve_sync_and_async_api_keys(self.openai_api_key)  # type: ignore[arg-type]
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
        _, async_api_key = _resolve_sync_and_async_api_keys(self.openai_api_key)  # type: ignore[arg-type]
        return openai.AsyncOpenAI(
            api_key=async_api_key,
            http_client=self.http_async_client,
            **self._client_params,
        )

    @property
    def _client_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            params["max_retries"] = self.max_retries
        return params

    @property
    @override
    def _provider(self) -> str:
        return "openai"

    @override
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
        return _parse_response(response)

    @override
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
        return _parse_response(response)

    def _payload(self, request: DecisionRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": to_decision_input(request["input"]),
            "questions": [
                _question_payload(name, question)
                for name, question in request["questions"].items()
            ],
        }


def _question_payload(name: str, question: Question) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": question.type,
        "name": name,
        "instructions": question.instructions,
    }
    if isinstance(question, Choice):
        payload["choices"] = [
            {"value": value}
            if description is None
            else {"value": value, "description": description}
            for value, description in question.choices.items()
        ]
    elif isinstance(question, Score):
        payload["levels"] = [
            level.model_dump(exclude_none=True) for level in question.as_levels()
        ]
    return payload


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
