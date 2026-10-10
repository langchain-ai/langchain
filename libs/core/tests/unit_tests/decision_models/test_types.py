from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from langchain_core.decision_models import (
    Answer,
    Choice,
    ChoiceAnswer,
    ChoiceProbability,
    DecisionOption,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    FakeDecisionModel,
    NoulAnswer,
    RefusalAnswer,
)


@pytest.mark.parametrize("value", [True, "0.5", -0.1, 1.1, math.nan, math.inf])
def test_invalid_probability(value: Any) -> None:
    with pytest.raises(ValidationError):
        NoulAnswer(probability=value)


@pytest.mark.parametrize("value", [True, "42", -1, 1.5])
def test_invalid_usage(value: Any) -> None:
    with pytest.raises(ValidationError):
        DecisionUsage(input_tokens=value)


def test_unknown_usage_is_not_zero() -> None:
    assert DecisionUsage().total_tokens is None
    assert DecisionUsage(input_tokens=42).total_tokens is None
    assert DecisionUsage(input_tokens=42, output_tokens=0).total_tokens == 42


def test_duplicate_options_rejected() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        Choice(instructions="Which?", options=[DecisionOption(value="x")] * 2)


def test_boolean_and_string_identity_round_trip() -> None:
    question = Choice(
        instructions="Which?",
        options=[DecisionOption(value=True), DecisionOption(value="true")],
    )
    restored = Choice.model_validate_json(question.model_dump_json())
    assert restored.options[0].value is True
    assert restored.options[1].value == "true"
    answer = ChoiceAnswer(
        value=True,
        probabilities=[
            ChoiceProbability(value=True, probability=0.7),
            ChoiceProbability(value="true", probability=0.3),
        ],
    )
    result = DecisionResponse(answers={"q": answer})
    restored_result = DecisionResponse.model_validate_json(result.model_dump_json())
    restored_answer = restored_result.answers["q"]
    assert isinstance(restored_answer, ChoiceAnswer)
    assert restored_answer.value is True
    assert restored_answer.selected_probability == 0.7


def test_request_schema_and_json_normalization(
    decision_request: DecisionRequest,
) -> None:
    adapter = TypeAdapter(DecisionRequest)
    restored = adapter.validate_json(adapter.dump_json(decision_request))
    assert restored == decision_request
    schema = adapter.json_schema()
    assert schema["required"] == ["state", "questions"]
    assert {"Noul", "Choice", "Score"} <= set(schema["$defs"])
    assert "Predicate" not in schema["$defs"]


def test_response_round_trip(response: DecisionResponse) -> None:
    assert DecisionResponse.model_validate_json(response.model_dump_json()) == response
    assert TypeAdapter(Answer).validate_python({"type": "refusal"}) == RefusalAnswer()


def test_public_input_output_schemas(response: DecisionResponse) -> None:
    model = FakeDecisionModel(response=response)
    assert model.get_input_jsonschema()["$defs"]["DecisionRequest"]["required"] == [
        "state",
        "questions",
    ]
    assert "answers" in model.get_output_jsonschema()["properties"]


def test_model_serialization_is_opt_in(response: DecisionResponse) -> None:
    model = FakeDecisionModel(response=response)
    assert not model.is_lc_serializable()
    assert model.to_json()["type"] == "not_implemented"
