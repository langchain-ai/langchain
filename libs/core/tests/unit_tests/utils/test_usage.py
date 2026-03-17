import operator

import pytest

from langchain_core.utils.usage import _dict_int_op


def test_dict_int_op_add() -> None:
    left = {"a": 1, "b": 2}
    right = {"b": 3, "c": 4}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 1, "b": 5, "c": 4}


def test_dict_int_op_subtract() -> None:
    left = {"a": 5, "b": 10}
    right = {"a": 2, "b": 3, "c": 1}
    result = _dict_int_op(left, right, lambda x, y: max(x - y, 0))
    assert result == {"a": 3, "b": 7, "c": 0}


def test_dict_int_op_nested() -> None:
    left = {"a": 1, "b": {"c": 2, "d": 3}}
    right = {"a": 2, "b": {"c": 1, "e": 4}}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 3, "b": {"c": 3, "d": 3, "e": 4}}


def test_dict_int_op_max_depth_exceeded() -> None:
    left = {"a": {"b": {"c": 1}}}
    right = {"a": {"b": {"c": 2}}}
    with pytest.raises(
        ValueError, match="max_depth=2 exceeded, unable to combine dicts"
    ):
        _dict_int_op(left, right, operator.add, max_depth=2)


def test_dict_int_op_float_add() -> None:
    """Test that float values are handled correctly."""
    left = {"tokens": 10, "cost": 0.05}
    right = {"tokens": 20, "cost": 0.03}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"tokens": 30, "cost": pytest.approx(0.08)}


def test_dict_int_op_float_subtract() -> None:
    """Test that float subtraction works correctly."""
    left = {"tokens": 10, "cost": 0.05}
    right = {"tokens": 3, "cost": 0.02}
    result = _dict_int_op(left, right, lambda x, y: max(x - y, 0))
    assert result == {"tokens": 7, "cost": pytest.approx(0.03)}


def test_dict_int_op_mixed_int_float() -> None:
    """Test that mixed int and float values are handled correctly."""
    left = {"tokens": 10, "cost": 0}  # cost is int (e.g., from cache init)
    right = {"tokens": 20, "cost": 0.005}  # cost is float (e.g., from provider)
    result = _dict_int_op(left, right, operator.add)
    assert result == {"tokens": 30, "cost": pytest.approx(0.005)}


def test_dict_int_op_nested_float() -> None:
    """Test that nested dicts with float values work correctly."""
    left = {"usage": {"tokens": 10, "cost": 0.05}}
    right = {"usage": {"tokens": 20, "cost": 0.03}}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"usage": {"tokens": 30, "cost": pytest.approx(0.08)}}


def test_dict_int_op_float_only_one_side() -> None:
    """Test float field present in only one dict."""
    left = {"tokens": 10, "cost": 0.05}
    right = {"tokens": 20}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"tokens": 30, "cost": pytest.approx(0.05)}


def test_dict_int_op_invalid_types() -> None:
    left = {"a": 1, "b": "string"}
    right = {"a": 2, "b": 3}
    with pytest.raises(
        ValueError,
        match="Only dict, int, and float values are supported",
    ):
        _dict_int_op(left, right, operator.add)
