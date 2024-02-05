import pytest

from langchain.evaluation.parsing.json_distance import JsonEditDistanceEvaluator


@pytest.fixture
def json_distance_evaluator() -> JsonEditDistanceEvaluator:
    return JsonEditDistanceEvaluator()


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_requires_input(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    assert json_distance_evaluator.requires_input is False


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_requires_reference(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    assert json_distance_evaluator.requires_reference is True


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_evaluation_name(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    assert json_distance_evaluator.evaluation_name == "json_edit_distance"


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_parse_json(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    string = '{"a": 1}'
    result = json_distance_evaluator._parse_json(string)
    assert result == {"a": 1}


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_evaluate_strings_simple_diff(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    prediction = '{"a":           1}'
    reference = '{"a": 2}'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    # Only 1 character flipped
    pytest.approx(1 / 7, result["score"])


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_evaluate_strings_complex_diff(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    prediction = '{"a":1, "b": {"c": 2, "d": 3}}'
    reference = '{"a": 1, "b": {"c": 2, "d": 4}}'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    # Only 1 character flipped
    pytest.approx(1 / len(reference.replace(" ", "")), result["score"])


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_evaluate_strings_list_diff(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 1, "b": 2}, {"a": 2, "b": 4}]'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    # Again only 1 character flipped
    pytest.approx(1 / len(reference.replace(" ", "")), result["score"])


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_evaluate_strings_list_same(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"b": 2, "a": 1}, {"b": 3, "a": 2}]'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == 0


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_evaluate_strings_list_diff_length(
    json_distance_evaluator: JsonEditDistanceEvaluator,
) -> None:
    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 1, "b": 2}]'
    result = json_distance_evaluator._evaluate_strings(
        prediction=prediction, reference=reference
    )
    pytest.approx(
        len('{"a":2,"b":3}') / len(reference.replace(" ", "")), result["score"]
    )


@pytest.mark.requires("rapidfuzz")
def test_json_distance_evaluator_evaluate_strings_custom_operator_equal() -> None:
    """Custom operator that returns 0.5 if strings are different."""

    def custom_distance(a: str, b: str) -> float:
        return 0.5 if a != b else 0.0

    evaluator = JsonEditDistanceEvaluator(string_distance=custom_distance)
    prediction = '{"a": "apple", "b": "banana"}'
    reference = '{"a": "apple", "b": "berries"}'
    result = evaluator._evaluate_strings(prediction=prediction, reference=reference)
    assert result["score"] == 0.5
