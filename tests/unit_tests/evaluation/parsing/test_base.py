import random

import pytest

from langchain.evaluation.parsing.base import (
    JsonEqualityEvaluator,
    JsonValidityEvaluator,
)


@pytest.fixture
def json_validity_evaluator() -> JsonValidityEvaluator:
    return JsonValidityEvaluator()


def test_json_validity_evaluator_requires_input(
    json_validity_evaluator: JsonValidityEvaluator,
) -> None:
    assert json_validity_evaluator.requires_input is False


def test_json_validity_evaluator_requires_reference(
    json_validity_evaluator: JsonValidityEvaluator,
) -> None:
    assert json_validity_evaluator.requires_reference is False


def test_json_validity_evaluator_evaluation_name(
    json_validity_evaluator: JsonValidityEvaluator,
) -> None:
    assert json_validity_evaluator.evaluation_name == "json_validity"


def test_json_validity_evaluator_evaluate_valid_json(
    json_validity_evaluator: JsonValidityEvaluator,
) -> None:
    prediction = '{"name": "John", "age": 30, "city": "New York"}'
    result = json_validity_evaluator.evaluate_strings(prediction=prediction)
    assert result == {"score": 1}


def test_json_validity_evaluator_evaluate_invalid_json(
    json_validity_evaluator: JsonValidityEvaluator,
) -> None:
    prediction = '{"name": "John", "age": 30, "city": "New York",}'
    result = json_validity_evaluator.evaluate_strings(prediction=prediction)
    assert result["score"] == 0
    assert result["reasoning"].startswith(
        "Expecting property name enclosed in double quotes"
    )


@pytest.fixture
def json_equality_evaluator() -> JsonEqualityEvaluator:
    return JsonEqualityEvaluator()


def test_json_equality_evaluator_requires_input(
    json_equality_evaluator: JsonEqualityEvaluator,
) -> None:
    assert json_equality_evaluator.requires_input is False


def test_json_equality_evaluator_requires_reference(
    json_equality_evaluator: JsonEqualityEvaluator,
) -> None:
    assert json_equality_evaluator.requires_reference is True


def test_json_equality_evaluator_evaluation_name(
    json_equality_evaluator: JsonEqualityEvaluator,
) -> None:
    assert json_equality_evaluator.evaluation_name == "json_equality"


def test_json_equality_evaluator_parse_json(
    json_equality_evaluator: JsonEqualityEvaluator,
) -> None:
    string = '{"a": 1}'
    result = json_equality_evaluator._parse_json(string)
    assert result == {"a": 1}


def test_json_equality_evaluator_evaluate_strings_equal(
    json_equality_evaluator: JsonEqualityEvaluator,
) -> None:
    prediction = '{"a": 1}'
    reference = '{"a": 1}'
    result = json_equality_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result == {"score": True}


def test_json_equality_evaluator_evaluate_strings_not_equal(
    json_equality_evaluator: JsonEqualityEvaluator,
) -> None:
    prediction = '{"a": 1}'
    reference = '{"a": 2}'
    result = json_equality_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result == {"score": False}


def test_json_equality_evaluator_evaluate_strings_custom_operator_equal() -> None:
    def operator(x: dict, y: dict) -> bool:
        return x["a"] == y["a"]

    evaluator = JsonEqualityEvaluator(operator=operator)
    prediction = '{"a": 1, "b": 2}'
    reference = '{"a": 1, "c": 3}'
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": True}


def test_json_equality_evaluator_evaluate_strings_custom_operator_not_equal() -> None:
    def operator(x: dict, y: dict) -> bool:
        return x["a"] == y["a"]

    evaluator = JsonEqualityEvaluator(operator=operator)
    prediction = '{"a": 1}'
    reference = '{"a": 2}'
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": False}


def test_json_equality_evaluator_evaluate_lists_permutation_invariant() -> None:
    evaluator = JsonEqualityEvaluator()
    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 2, "b": 3}, {"a": 1, "b": 2}]'
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": True}

    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 2, "b": 3}, {"a": 1, "b": 4}]'
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": False}

    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 2, "b": 3}]'
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": False}

    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 2, "b": 3}, {"a": 1, "b": 2}, {"a": 3, "b": 4}]'
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": False}

    prediction = '[{"a": 1, "b": 2}, {"a": 2, "b": 3}]'
    reference = '[{"a": 2, "b": 3}, {"b": 2,"a": 1}, {"a": 3, "b": 4}]'
    result = evaluator.evaluate_strings(prediction=reference, reference=prediction)
    assert result == {"score": False}

    # Limit tests
    prediction = (
        "[" + ",".join([f'{{"a": {i}, "b": {i+1}}}' for i in range(1000)]) + "]"
    )
    rlist = [f'{{"a": {i}, "b": {i+1}}}' for i in range(1000)]
    random.shuffle(rlist)
    reference = "[" + ",".join(rlist) + "]"
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": True}

    prediction = (
        "[" + ",".join([f'{{"b": {i+1}, "a": {i}}}' for i in range(1000)]) + "]"
    )
    reference = (
        "["
        + ",".join(
            [f'{{"a": {i+1}, "b": {i+2}}}' for i in range(999)]
            + ['{"a": 1000, "b": 1001}']
        )
        + "]"
    )
    result = evaluator.evaluate_strings(prediction=prediction, reference=reference)
    assert result == {"score": False}
