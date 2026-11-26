"""Unit tests for decision model types."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from langchain_core.decisions import (
    Box,
    BoxAnswer,
    Choice,
    ClassifierResponse,
    Noul,
    NoulCriteria,
    Score,
)


@pytest.mark.parametrize(
    ("question_type", "kwargs"),
    [
        (Noul, {}),
        (Choice, {"criteria": {"a": None}}),
        (Score, {"criteria": ["low", "high"]}),
        (Box, {}),
    ],
)
def test_questions_require_instructions(
    question_type: type[Noul | Choice | Score | Box], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="instructions"):
        question_type(**kwargs)


def test_noul_criteria_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        Noul.model_validate(
            {"instructions": "Is this urgent?", "criteria": {"maybe": "Unsure."}}
        )


@pytest.mark.parametrize(
    ("question_type", "kwargs"),
    [
        (Choice, {"criteria": {}}),
        (Score, {"criteria": []}),
        (Score, {"criteria": ["only level"]}),
        (Choice, {"criteria": {"a": None}, "extra": 1}),
    ],
)
def test_questions_validate_their_criteria(
    question_type: type[Choice | Score], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        question_type(instructions="Classify.", **kwargs)


def test_questions_accept_structured_content() -> None:
    question = Choice(
        instructions={"task": "Route the request."},
        criteria={"billing": {"examples": ["refund"]}, "other": None},
    )

    assert question.model_dump(mode="json") == {
        "type": "choice",
        "criteria": {"billing": {"examples": ["refund"]}, "other": None},
        "instructions": {"task": "Route the request."},
    }


def test_criteria_accept_any_json_value() -> None:
    noul = Noul(instructions="Is this urgent?", criteria=NoulCriteria(true=1, false=0))
    score = Score(instructions="Rate it.", criteria=[0.0, {"label": "high"}])

    assert noul.criteria == NoulCriteria(true=1, false=0)
    assert score.criteria == [0.0, {"label": "high"}]


def test_response_views_filter_by_answer_type() -> None:
    response = ClassifierResponse.model_validate(
        {
            "model": "fake-model",
            "answers": {
                "urgent": {"type": "noul", "noul": 0.9},
                "dept": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 1.0},
                    "confidence": 1.0,
                },
                "severity": {
                    "type": "score",
                    "score": 0.2,
                    "legend": {"0": "low", "1": {"label": "high"}},
                    "probabilities": {"0": 0.8, "1": 0.2},
                    "confidence": 0.5,
                },
                "dog": {
                    "type": "box",
                    "box": {"top": 0.1, "bottom": 0.9, "left": 0.2, "right": 0.8},
                    "confidence": None,
                },
            },
        }
    )

    assert set(response.nouls) == {"urgent"}
    assert set(response.choices) == {"dept"}
    assert response.scores["severity"].legend == {0: "low", 1: {"label": "high"}}
    assert response.boxes["dog"] == BoxAnswer.model_validate(
        {
            "type": "box",
            "box": {"top": 0.1, "bottom": 0.9, "left": 0.2, "right": 0.8},
        }
    )


def test_response_rejects_unknown_answer_types() -> None:
    with pytest.raises(ValidationError):
        ClassifierResponse.model_validate(
            {
                "model": "fake-model",
                "answers": {
                    "urgent": {"type": "noul", "noul": 0.9},
                    "future": {"type": "UNKNOWN"},
                },
            }
        )


def test_response_ignores_unknown_fields() -> None:
    response = ClassifierResponse.model_validate(
        {"model": "fake-model", "answers": {}, "prompt_feedback": None}
    )

    assert response.answers == {}
    assert response.usage is None


def test_partial_usage_is_completed() -> None:
    response = ClassifierResponse.model_validate(
        {"model": "fake-model", "answers": {}, "usage": {"input_tokens": 12}}
    )

    assert response.usage == {
        "input_tokens": 12,
        "output_tokens": 0,
        "total_tokens": 12,
    }


def test_usage_token_details_are_kept() -> None:
    response = ClassifierResponse.model_validate(
        {
            "model": "fake-model",
            "answers": {},
            "usage": {
                "input_tokens": 12,
                "output_tokens": 3,
                "input_token_details": {"cache_read": 8},
            },
        }
    )

    assert response.usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "input_token_details": {"cache_read": 8},
    }
