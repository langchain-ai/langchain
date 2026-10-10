"""Private adapter from core decisions to the existing OpenAI transport."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.decision_models import (
    BaseDecisionModel,
    Choice,
    ChoiceAnswer,
    ChoiceProbability,
    DecisionModelProfile,
    DecisionRequest,
    DecisionResponse,
    DecisionResponseValidationError,
    DecisionUsage,
    Noul,
    NoulAnswer,
    RefusalAnswer,
    Score,
    ScoreAnswer,
)
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.messages import BaseMessage, convert_to_openai_messages
from pydantic import Field
from typing_extensions import override

from langchain_openai.decisions._input import to_decision_input
from langchain_openai.decisions.base import OpenAIDecisions

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.decision_models import Answer, Question
    from langchain_core.runnables import RunnableConfig

    from langchain_openai._compat import httpx
    from langchain_openai.decisions.types import State as NativeState


def _check_message_parts(state: Any) -> None:
    # The native JSON serializer can retain unsupported blocks as ordinary text.
    # The canonical interface must reject them rather than silently lose evidence.
    if isinstance(state, BaseMessage):
        content = convert_to_openai_messages(state)["content"]
        if isinstance(content, list) and any(
            not isinstance(block, dict)
            or block.get("type") not in {"text", "image_url"}
            or (
                block.get("type") == "image_url"
                and not block.get("image_url", {}).get("url", "").startswith("data:")
            )
            for block in content
        ):
            msg = "Canonical OpenAI decisions support text and embedded images only."
            raise ModelInvalidRequestError(msg)
    elif isinstance(state, dict):
        for value in state.values():
            _check_message_parts(value)
    elif isinstance(state, list):
        for value in state:
            _check_message_parts(value)


def _payload(native: OpenAIDecisions, request: DecisionRequest) -> dict[str, Any]:
    _check_message_parts(request["state"])
    questions: list[dict[str, Any]] = []
    for name, question in request["questions"].items():
        item: dict[str, Any] = {"name": name, "instructions": question.instructions}
        if isinstance(question, Noul):
            item["type"] = "predicate"
        elif isinstance(question, Choice):
            item.update(
                type="choice",
                choices=[
                    option.model_dump(exclude_none=True) for option in question.options
                ],
            )
        else:
            item.update(
                type="score",
                levels=[
                    level.model_dump(exclude_none=True) for level in question.levels
                ],
            )
        questions.append(item)
    return {
        "model": native.model,
        "input": to_decision_input(cast("NativeState", request["state"])),
        "questions": questions,
    }


def _answer(raw: dict[str, Any], question: Question) -> Answer:
    kind = raw["type"]
    extras = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "name",
            "type",
            "probability",
            "choice",
            "probabilities",
            "score",
            "confidence",
            "abstained",
        }
    }
    common: dict[str, Any] = {
        "response_metadata": extras,
        "abstained": raw.get("abstained"),
    }
    if kind == "refusal":
        return RefusalAnswer(**common)
    if kind == "predicate" and isinstance(question, Noul):
        return NoulAnswer(probability=raw["probability"], **common)
    if kind == "choice" and isinstance(question, Choice):
        probabilities = raw["probabilities"]
        indexed = {(type(item["value"]), item["value"]): item for item in probabilities}
        expected = [(type(option.value), option.value) for option in question.options]
        if len(indexed) != len(probabilities) or set(indexed) != set(expected):
            msg = "OpenAI choice probabilities do not cover the requested options."
            raise DecisionResponseValidationError(msg)
        return ChoiceAnswer(
            value=raw["choice"],
            probabilities=[
                ChoiceProbability.model_validate(indexed[key]) for key in expected
            ],
            provider_confidence=raw.get("confidence"),
            **common,
        )
    if kind == "score" and isinstance(question, Score):
        probabilities = raw["probabilities"]
        indexed_levels = {item["value"]: item for item in probabilities}
        if any(type(item["value"]) is not int for item in probabilities) or (
            len(indexed_levels) != len(probabilities)
            or set(indexed_levels) != set(range(len(question.levels)))
            or any(
                indexed_levels[i]["label"] != level.label
                for i, level in enumerate(question.levels)
            )
        ):
            msg = "OpenAI score probabilities do not cover the requested rubric."
            raise DecisionResponseValidationError(msg)
        return ScoreAnswer(
            score=raw["score"],
            levels=question.levels,
            probabilities=[
                indexed_levels[i]["probability"] for i in range(len(question.levels))
            ],
            provider_confidence=raw.get("confidence"),
            **common,
        )
    msg = "OpenAI decision answer type does not match its question."
    raise DecisionResponseValidationError(msg)


def _translate(response: httpx.Response, request: DecisionRequest) -> DecisionResponse:
    try:
        body = response.json()
        answers: dict[str, Any] = {}
        for raw in body["answers"]:
            name = raw["name"]
            if name in answers or name not in request["questions"]:
                msg = "OpenAI decision answer IDs must be unique and requested."
                raise DecisionResponseValidationError(msg)
            answers[name] = _answer(raw, request["questions"][name])
        usage = body.get("usage") or {}
        metadata = {
            key: value
            for key, value in body.items()
            if key not in {"answers", "model", "usage"}
        }
        metadata.update(provider="openai", provider_usage=usage)
        headers = {
            key: value
            for key, value in response.headers.items()
            if key
            in {
                "x-request-id",
                "x-litellm-call-id",
                "x-litellm-response-cost",
                "x-langsmith-request-id",
            }
        }
        metadata["response_headers"] = headers
        if "x-request-id" in headers:
            metadata.setdefault("request_id", headers["x-request-id"])
        return DecisionResponse(
            model=body.get("model"),
            answers=answers,
            usage=DecisionUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            ),
            response_metadata=metadata,
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        msg = "Invalid canonical response from the OpenAI Decisions API."
        raise DecisionResponseValidationError(msg) from None


class _OpenAIDecisionModel(BaseDecisionModel):
    native: OpenAIDecisions = Field(exclude=True, repr=False)
    profile: DecisionModelProfile | None = Field(
        default_factory=lambda: DecisionModelProfile(
            question_types=["noul", "choice", "score"],
            state_types=["text", "json", "messages"],
            boolean_choices=True,
        )
    )

    @property
    @override
    def _tracing_metadata(self) -> dict[str, Any]:
        return {
            "ls_provider": "openai",
            "ls_model_name": self.native.model,
            "ls_model_type": "chat",
            "decision_model": True,
        }

    @override
    def _decide(
        self,
        request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun,
    ) -> DecisionResponse:
        response = self.native._post_decision(_payload(self.native, request))  # noqa: SLF001
        return _translate(response, request)

    @override
    async def _adecide(
        self,
        request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> DecisionResponse:
        response = await self.native._apost_decision(_payload(self.native, request))  # noqa: SLF001
        return _translate(response, request)
