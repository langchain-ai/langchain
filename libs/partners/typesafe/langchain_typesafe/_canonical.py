"""Private adapter from core decisions to the existing TypeSafe transport."""

from __future__ import annotations

import json
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
from langchain_core.messages import BaseMessage
from pydantic import Field
from typing_extensions import override

from langchain_typesafe._state import serialize_state
from langchain_typesafe.classifier import TypeSafeClassifier  # noqa: TC001
from langchain_typesafe.client import parse_response

if TYPE_CHECKING:
    import httpx2
    from langchain_core.callbacks import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.decision_models import Answer, DecisionLevel, Question
    from langchain_core.runnables import RunnableConfig

    from langchain_typesafe.types import State as NativeState


def _check_text_messages(state: Any) -> None:
    if isinstance(state, BaseMessage):
        if isinstance(state.content, list) and any(
            isinstance(block, dict) and block.get("type") != "text"
            for block in state.content
        ):
            msg = "Canonical TypeSafe decisions do not support message media."
            raise ModelInvalidRequestError(msg)
    elif isinstance(state, dict):
        for value in state.values():
            _check_text_messages(value)
    elif isinstance(state, list):
        for value in state:
            _check_text_messages(value)


def _criterion(level: DecisionLevel) -> str:
    return (
        level.label
        if level.description is None
        else f"{level.label}: {level.description}"
    )


def _payload(native: TypeSafeClassifier, request: DecisionRequest) -> dict[str, Any]:
    _check_text_messages(request["state"])
    questions: dict[str, Any] = {}
    for name, question in request["questions"].items():
        item: dict[str, Any] = {
            "type": question.type,
            "instructions": question.instructions,
        }
        if isinstance(question, Choice):
            item["criteria"] = {
                option.value: option.description for option in question.options
            }
        elif isinstance(question, Score):
            item["criteria"] = [_criterion(level) for level in question.levels]
        questions[name] = item
    return {
        "model": native.model,
        "state": serialize_state(cast("NativeState", request["state"])),
        "questions": questions,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = "TypeSafe response contains duplicate object keys."
            raise DecisionResponseValidationError(msg)
        result[key] = value
    return result


def _answer(raw: dict[str, Any], question: Question) -> Answer:
    extras = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "type",
            "noul",
            "choice",
            "probabilities",
            "score",
            "legend",
            "confidence",
            "abstained",
        }
    }
    common: dict[str, Any] = {
        "response_metadata": extras,
        "abstained": raw.get("abstained"),
    }
    kind = raw["type"]
    if kind == "refusal":
        return RefusalAnswer(**common)
    if kind == "noul" and isinstance(question, Noul):
        return NoulAnswer(probability=raw["noul"], **common)
    if kind == "choice" and isinstance(question, Choice):
        probabilities = raw["probabilities"]
        if set(probabilities) != {option.value for option in question.options}:
            msg = "TypeSafe probabilities do not cover the requested options."
            raise DecisionResponseValidationError(msg)
        return ChoiceAnswer(
            value=raw["choice"],
            probabilities=[
                ChoiceProbability(
                    value=option.value, probability=probabilities[option.value]
                )
                for option in question.options
            ],
            provider_confidence=raw.get("confidence"),
            **common,
        )
    if kind == "score" and isinstance(question, Score):
        probabilities = raw["probabilities"]
        expected = {
            str(i): _criterion(level) for i, level in enumerate(question.levels)
        }
        if set(probabilities) != set(expected) or raw["legend"] != expected:
            msg = "TypeSafe score probabilities do not cover the requested rubric."
            raise DecisionResponseValidationError(msg)
        common["response_metadata"]["provider_legend"] = raw["legend"]
        return ScoreAnswer(
            score=raw["score"],
            levels=question.levels,
            probabilities=[probabilities[str(i)] for i in range(len(question.levels))],
            provider_confidence=raw.get("confidence"),
            **common,
        )
    msg = "TypeSafe decision answer type does not match its question."
    raise DecisionResponseValidationError(msg)


def _translate(response: httpx2.Response, request: DecisionRequest) -> DecisionResponse:
    if not response.is_success:
        parse_response(response)
    try:
        body = json.loads(response.content, object_pairs_hook=_unique_object)
        answers = {
            name: _answer(raw, request["questions"][name])
            for name, raw in body["answers"].items()
        }
        usage = body.get("usage") or {}
        metadata = {
            key: value
            for key, value in body.items()
            if key not in {"answers", "model", "usage"}
        }
        metadata.update(provider="typesafe", provider_usage=usage)
        headers = {
            key: value
            for key, value in response.headers.items()
            if key
            in {
                "x-typesafe-request-id",
                "x-litellm-call-id",
                "x-litellm-response-cost",
                "x-langsmith-request-id",
            }
        }
        metadata["response_headers"] = headers
        if "x-typesafe-request-id" in headers:
            metadata.setdefault("request_id", headers["x-typesafe-request-id"])
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
        msg = "Invalid canonical response from the TypeSafe API."
        raise DecisionResponseValidationError(msg) from None


class _TypeSafeDecisionModel(BaseDecisionModel):
    native: TypeSafeClassifier = Field(exclude=True, repr=False)
    profile: DecisionModelProfile | None = Field(
        default_factory=lambda: DecisionModelProfile(
            question_types=["noul", "choice", "score"],
            state_types=["text", "json", "messages"],
            boolean_choices=False,
        )
    )

    @property
    @override
    def _tracing_metadata(self) -> dict[str, Any]:
        return {
            "ls_provider": "typesafe",
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
        response = self.native._post_classification(_payload(self.native, request))  # noqa: SLF001
        return _translate(response, request)

    @override
    async def _adecide(
        self,
        request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> DecisionResponse:
        response = await self.native._apost_classification(  # noqa: SLF001
            _payload(self.native, request)
        )
        return _translate(response, request)
