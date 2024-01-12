import pytest

from langchain.evaluation.parsing.json_schema import JsonSchemaEvaluator


@pytest.fixture
def json_schema_evaluator() -> JsonSchemaEvaluator:
    return JsonSchemaEvaluator()


@pytest.mark.requires("jsonschema")
def test_json_schema_evaluator_requires_input(
    json_schema_evaluator: JsonSchemaEvaluator,
) -> None:
    assert json_schema_evaluator.requires_input is False


@pytest.mark.requires("jsonschema")
def test_json_schema_evaluator_requires_reference(
    json_schema_evaluator: JsonSchemaEvaluator,
) -> None:
    assert json_schema_evaluator.requires_reference is True


@pytest.mark.requires("jsonschema")
def test_json_schema_evaluator_evaluation_name(
    json_schema_evaluator: JsonSchemaEvaluator,
) -> None:
    assert json_schema_evaluator.evaluation_name == "json_schema_validation"


@pytest.mark.requires("jsonschema")
def test_json_schema_evaluator_valid_prediction(
    json_schema_evaluator: JsonSchemaEvaluator,
) -> None:
    prediction = '{"name": "John", "age": 30}'
    reference = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    result = json_schema_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] is True


@pytest.mark.requires("jsonschema")
def test_json_schema_evaluator_invalid_prediction(
    json_schema_evaluator: JsonSchemaEvaluator,
) -> None:
    prediction = '{"name": "John", "age": "30"}'  # age is a string instead of integer
    reference = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    result = json_schema_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] is False
    assert "reasoning" in result


@pytest.mark.requires("jsonschema")
def test_json_schema_evaluator_missing_property(
    json_schema_evaluator: JsonSchemaEvaluator,
) -> None:
    prediction = '{"name": "John"}'  # age property is missing
    reference = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    result = json_schema_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] is False
    assert "reasoning" in result
