from __future__ import annotations

import pytest

from langchain_core.decision_models import (
    Choice,
    ChoiceAnswer,
    ChoiceProbability,
    DecisionLevel,
    DecisionOption,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)


@pytest.fixture
def decision_request() -> DecisionRequest:
    return {
        "state": "Payments have failed for three days. Please help!",
        "questions": {
            "urgent": Noul(instructions="Is this urgent?"),
            "team": Choice(
                instructions="Which team?",
                options=[
                    DecisionOption(value="billing"),
                    DecisionOption(value="technical"),
                ],
            ),
            "severity": Score(
                instructions="How severe?",
                levels=[
                    DecisionLevel(label="calm"),
                    DecisionLevel(label="frustrated"),
                    DecisionLevel(label="angry"),
                ],
            ),
        },
    }


@pytest.fixture
def response(decision_request: DecisionRequest) -> DecisionResponse:
    question = decision_request["questions"]["severity"]
    assert isinstance(question, Score)
    return DecisionResponse(
        model="test-model",
        answers={
            "urgent": NoulAnswer(probability=0.95),
            "team": ChoiceAnswer(
                value="technical",
                probabilities=[
                    ChoiceProbability(value="billing", probability=0.1),
                    ChoiceProbability(value="technical", probability=0.9),
                ],
                provider_confidence=0.8,
            ),
            "severity": ScoreAnswer(
                score=1.25,
                levels=question.levels,
                probabilities=[0.1, 0.55, 0.35],
                provider_confidence=0.7,
            ),
        },
        usage=DecisionUsage(input_tokens=42, output_tokens=0),
    )
