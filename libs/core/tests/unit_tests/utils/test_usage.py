import operator
from typing import cast

import pytest

from langchain_core.messages.ai import UsageMetadata
from langchain_core.utils.usage import _dict_int_op, get_billable_input_tokens


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
        ValueError, match="max_depth=2 exceeded, unable to combine dicts."
    ):
        _dict_int_op(left, right, operator.add, max_depth=2)


def test_dict_int_op_invalid_types() -> None:
    left = {"a": 1, "b": "string"}
    right = {"a": 2, "b": 3}
    with pytest.raises(
        ValueError,
        match="Only dict and int values are supported.",
    ):
        _dict_int_op(left, right, operator.add)


def test_get_billable_input_tokens_basic() -> None:
    """Test basic billable token calculation."""
    usage_metadata = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_tokens": 1500,
    }

    # Without input_token_details, should return full input_tokens
    # Note: cast() is used to convert plain dict to UsageMetadata type for testing
    # Done to avoid circular imports
    result = get_billable_input_tokens(cast("UsageMetadata", usage_metadata))
    assert result == 1000

    # With cache usage
    cache_usage_metadata = {
        "input_tokens": 151998,
        "output_tokens": 691,
        "total_tokens": 152689,
        "input_token_details": {
            "cache_creation": 0,
            "cache_read": 151995,
        },
    }

    # Should subtract cached tokens from total
    result = get_billable_input_tokens(cast("UsageMetadata", cache_usage_metadata))
    assert result == 3  # 151998 - 0 - 151995


def test_get_billable_input_tokens_with_cache_creation() -> None:
    """Test billable token calculation with cache creation."""
    usage_metadata = {
        "input_tokens": 10000,
        "output_tokens": 500,
        "total_tokens": 10500,
        "input_token_details": {
            "cache_creation": 5000,
            "cache_read": 2000,
        },
    }

    # Should subtract both cache_creation and cache_read
    result = get_billable_input_tokens(cast("UsageMetadata", usage_metadata))
    assert result == 3000  # 10000 - 5000 - 2000


def test_get_billable_input_tokens_partial_details() -> None:
    """Test with only some cache details present."""
    usage_metadata = {
        "input_tokens": 5000,
        "output_tokens": 300,
        "total_tokens": 5300,
        "input_token_details": {
            "cache_read": 1000,
            # cache_creation missing - should default to 0
        },
    }

    result = get_billable_input_tokens(cast("UsageMetadata", usage_metadata))
    assert result == 4000  # 5000 - 1000 - 0


def test_get_billable_input_tokens_empty_details() -> None:
    """Test with empty input_token_details."""
    usage_metadata = {
        "input_tokens": 2000,
        "output_tokens": 400,
        "total_tokens": 2400,
        "input_token_details": {},
    }

    result = get_billable_input_tokens(cast("UsageMetadata", usage_metadata))
    assert result == 2000  # No cache usage, return full amount
