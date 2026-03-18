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


def test_dict_int_op_invalid_types() -> None:
    left = {"a": 1, "b": "string"}
    right = {"a": 2, "b": 3}
    with pytest.raises(
        ValueError,
        match="Only dict, int, and float values are supported",
    ):
        _dict_int_op(left, right, operator.add)


def test_dict_int_op_float_add() -> None:
    """Test that float values are handled correctly."""
    left = {"a": 0.5, "b": 1.2}
    right = {"b": 0.3, "c": 2.5}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 0.5, "b": 1.5, "c": 2.5}


def test_dict_int_op_mixed_int_float() -> None:
    """Test that mixed int and float values are handled correctly."""
    left = {"a": 1, "b": 0.5}
    right = {"a": 2.5, "b": 3}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 3.5, "b": 3.5}


def test_dict_int_op_nested_float() -> None:
    """Test that nested dictionaries with float values are handled correctly."""
    left = {"a": 1, "b": {"c": 0.5, "d": 3}}
    right = {"a": 2, "b": {"c": 1.5, "e": 0.1}}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 3, "b": {"c": 2.0, "d": 3, "e": 0.1}}
