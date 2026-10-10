from __future__ import annotations

from typing import Any

import pytest
from langchain_core.decision_models import (
    BaseDecisionModel,
    ChoiceAnswer,
    ChoiceProbability,
    DecisionLevel,
    DecisionRequest,
    DecisionResponse,
    FakeDecisionModel,
    Noul,
    NoulAnswer,
    ScoreAnswer,
)

from langchain_tests.unit_tests.decision_models import DecisionModelUnitTests


def _response() -> DecisionResponse:
    return DecisionResponse(
        answers={
            "urgent": NoulAnswer(probability=0.95),
            "team": ChoiceAnswer(
                value="technical",
                probabilities=[
                    ChoiceProbability(value="billing", probability=0.1),
                    ChoiceProbability(value="technical", probability=0.9),
                ],
            ),
            "severity": ScoreAnswer(
                score=1.25,
                levels=[
                    DecisionLevel(label="calm"),
                    DecisionLevel(label="frustrated"),
                    DecisionLevel(label="angry"),
                ],
                probabilities=[0.1, 0.55, 0.35],
            ),
        }
    )


class TestFakeDecisionModel(DecisionModelUnitTests):
    @property
    def decision_model_class(self) -> type[BaseDecisionModel]:
        return FakeDecisionModel

    @property
    def decision_model_params(self) -> dict[str, Any]:
        return {"response": _response()}


def test_conformance_detects_missing_answers() -> None:
    suite = TestFakeDecisionModel()
    request = DecisionRequest(state="hello", questions={})
    response = _response()
    request["questions"] = {"urgent": Noul(instructions="Urgent?")}
    response.answers.clear()
    with pytest.raises(AssertionError):
        suite._assert_response(request, response)
