import pytest

from langchain.evaluation.parsing.json_distance import JsonDistanceEvaluator


@pytest.fixture
def json_distance_evaluator() -> JsonDistanceEvaluator:
    return JsonDistanceEvaluator()


def test_json_distance_evaluator_requires_input(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    assert json_distance_evaluator.requires_input is False


def test_json_distance_evaluator_requires_reference(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    assert json_distance_evaluator.requires_reference is True


def test_json_distance_evaluator_evaluation_name(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    assert json_distance_evaluator.evaluation_name == "json_distance"


def test_json_distance_evaluator_parse_json(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    string = '{"a": 1}'
    result = json_distance_evaluator._parse_json(string)
    assert result == {"a": 1}


def test_json_distance_evaluator_evaluate_strings_simple_diff(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    prediction = '{"a": 1}'
    reference = '{"a": 2}'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result == {"score": 1.0}


def test_json_distance_evaluator_evaluate_strings_complex_diff(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    prediction = '{"a": 1, "b": {"c": 2, "d": 3}}'
    reference = '{"a": 1, "b": {"c": 2, "d": 4}}'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result == {"score": 1.0}


def test_json_distance_evaluator_evaluate_strings_list_diff(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 1, "b": 2}, {"a": 2, "b": 4}]'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result == {"score": 1.0}


def test_json_distance_evaluator_evaluate_strings_list_same(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 2, "b": 3}, {"a": 1, "b": 2}]'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result == {"score": 1.0}


def test_json_distance_evaluator_evaluate_strings_list_diff_length(
    json_distance_evaluator: JsonDistanceEvaluator,
) -> None:
    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 1, "b": 2}]'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result == {"score": 1.0}


def test_json_distance_evaluator_evaluate_strings_custom_operator_equal() -> None:
    """Custom operator that returns 0.5 if strings are different."""

    def custom_distance(a: str, b: str) -> float:
        return 0.5 if a != b else 0.0

    evaluator = JsonDistanceEvaluator(string_distance=custom_distance)
    prediction = '{"a": "apple", "b": "banana"}'
    reference = '{"a": "apple", "b": "berries"}'
    result = evaluator._evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": 0.5}
