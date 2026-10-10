"""Private validation of complete decision requests and responses."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError

from langchain_core.decision_models.types import (
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    RefusalAnswer,
    Score,
    ScoreAnswer,
)
from langchain_core.exceptions import ModelError, ModelInvalidRequestError
from langchain_core.messages import BaseMessage

if TYPE_CHECKING:
    from langchain_core.decision_models.types import DecisionModelProfile

_REQUEST = TypeAdapter(DecisionRequest)
# Permit rounded provider probabilities without silently renormalizing them.
_ROUNDING_TOLERANCE = 1e-3


class DecisionResponseValidationError(ModelError):
    """A response violates the requested decision contract.

    This error is not retryable by classification. Runnable `with_retry` still uses
    the caller's exception predicate, so configure retry types explicitly.
    """

    is_retryable = False


def _validate_state(value: Any) -> None:
    if isinstance(value, BaseMessage):
        _validate_state(value.content)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = "Decision state object keys must be strings."
                raise ModelInvalidRequestError(msg)
            _validate_state(item)
    elif isinstance(value, list):
        for item in value:
            _validate_state(item)
    elif isinstance(value, float) and not math.isfinite(value):
        msg = "Decision state must contain only finite JSON numbers."
        raise ModelInvalidRequestError(msg)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        msg = "Decision state must be JSON-compatible or contain messages."
        raise ModelInvalidRequestError(msg)


def normalize_request(request: DecisionRequest) -> DecisionRequest:
    if not isinstance(request, dict) or set(request) != {"state", "questions"}:
        msg = "A decision request must contain exactly state and questions."
        raise ModelInvalidRequestError(msg)
    try:
        _validate_state(request["state"])
        normalized = _REQUEST.validate_python(request, strict=True)
    except (ValidationError, TypeError, RecursionError):
        msg = "Invalid decision state or question definitions."
        raise ModelInvalidRequestError(msg) from None
    if any(not name.strip() for name in normalized["questions"]):
        msg = "Decision question IDs must not be empty."
        raise ModelInvalidRequestError(msg)
    return normalized


def validate_capabilities(
    request: DecisionRequest, profile: DecisionModelProfile | None
) -> None:
    if profile is None:
        return
    state = request["state"]
    state_type = (
        "text"
        if isinstance(state, str)
        else "messages"
        if isinstance(state, BaseMessage)
        or (isinstance(state, list) and any(isinstance(x, BaseMessage) for x in state))
        else "json"
    )
    if "state_types" in profile and state_type not in profile["state_types"]:
        msg = "Decision model does not support this state form."
        raise ModelInvalidRequestError(msg)
    if len(request["questions"]) > profile.get("max_questions", math.inf):
        msg = "Decision request exceeds the model's question limit."
        raise ModelInvalidRequestError(msg)
    for question in request["questions"].values():
        if (
            "question_types" in profile
            and question.type not in profile["question_types"]
        ):
            msg = "Decision model does not support this question type."
            raise ModelInvalidRequestError(msg)
        if isinstance(question, Choice):
            if profile.get("boolean_choices") is False and any(
                isinstance(option.value, bool) for option in question.options
            ):
                msg = "Decision model does not support boolean choice values."
                raise ModelInvalidRequestError(msg)
            if len(question.options) > profile.get("max_options", math.inf):
                msg = "Decision question exceeds the model's option limit."
                raise ModelInvalidRequestError(msg)
        elif isinstance(question, Score) and len(question.levels) > profile.get(
            "max_options", math.inf
        ):
            msg = "Decision question exceeds the model's level limit."
            raise ModelInvalidRequestError(msg)


def _invalid(message: str) -> None:
    raise DecisionResponseValidationError(message)


def _validate_mass(probabilities: list[float]) -> None:
    if not math.isclose(sum(probabilities), 1, rel_tol=0, abs_tol=_ROUNDING_TOLERANCE):
        _invalid("Decision probabilities must sum to one.")


def validate_response(
    request: DecisionRequest, response: DecisionResponse
) -> DecisionResponse:
    try:
        response = DecisionResponse.model_validate(response)
    except ValidationError:
        _invalid("Invalid decision response fields.")
    if set(response.answers) != set(request["questions"]):
        _invalid("Decision answer IDs must match the requested question IDs exactly.")
    for name, question in request["questions"].items():
        answer = response.answers[name]
        if isinstance(answer, RefusalAnswer):
            continue
        if answer.type != question.type:
            _invalid("Decision answer type does not match its question.")
        if isinstance(question, Choice) and isinstance(answer, ChoiceAnswer):
            expected = [
                (type(option.value), option.value) for option in question.options
            ]
            actual = [(type(item.value), item.value) for item in answer.probabilities]
            if actual != expected or (type(answer.value), answer.value) not in expected:
                _invalid(
                    "Decision choice values must match the offered options in order."
                )
            _validate_mass([item.probability for item in answer.probabilities])
        elif isinstance(question, Score) and isinstance(answer, ScoreAnswer):
            if answer.levels != question.levels or len(answer.probabilities) != len(
                question.levels
            ):
                _invalid("Decision score levels must match the requested rubric.")
            _validate_mass(answer.probabilities)
            if not 0 <= answer.score <= len(question.levels) - 1:
                _invalid("Decision score must remain within the requested rubric.")
            mean = sum(i * p for i, p in enumerate(answer.probabilities))
            if not math.isclose(
                answer.score, mean, rel_tol=0, abs_tol=_ROUNDING_TOLERANCE
            ):
                _invalid("Decision score must equal the expected ordinal index.")
    return response
