"""Documentation-derived cases; these are not live integration results."""

from __future__ import annotations

import pytest

from langchain_core.decision_models import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    FakeDecisionModel,
)
from langchain_core.exceptions import ModelInvalidRequestError


def test_scaledown_confidence_cost_and_state(
    decision_request: DecisionRequest,
    response: DecisionResponse,
) -> None:
    # https://docs.scaledown.ai/api-reference/openapi.json
    answer = response.answers["team"]
    assert isinstance(answer, ChoiceAnswer)
    answer.provider_confidence = 0.9  # Top probability, rather than Jev's rescaling.
    response.response_metadata = {
        "provider": "scaledown",
        "provider_usage": {"cost": 0.001},
    }
    model = FakeDecisionModel(response=response, profile={"state_types": ["text"]})
    result = model.invoke(decision_request)
    choice = result.answers["team"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.provider_confidence == choice.selected_probability == 0.9
    assert result.response_metadata["provider_usage"]["cost"] == 0.001
    with pytest.raises(ModelInvalidRequestError):
        model.invoke({**decision_request, "state": {"unsupported": True}})


def test_laya_confidence_and_abstention(
    decision_request: DecisionRequest,
    response: DecisionResponse,
) -> None:
    # https://github.com/NandhaKishorM/laya#decision-primitives
    answer = response.answers["team"]
    assert isinstance(answer, ChoiceAnswer)
    answer.provider_confidence = 0.12  # Illustrative, not a measured entropy value.
    answer.abstained = True
    answer.response_metadata = {"confidence_method": "normalized_entropy"}
    choice = (
        FakeDecisionModel(response=response).invoke(decision_request).answers["team"]
    )
    assert isinstance(choice, ChoiceAnswer)
    assert choice.provider_confidence == 0.12
    assert choice.selected_probability == 0.9
    assert choice.abstained is True
