"""Usage utilities."""

from collections.abc import Callable
from typing import Any


def _dict_int_op(
    left: dict,
    right: dict,
    op: Callable[[int, int], int],
    *,
    default: int = 0,
    depth: int = 0,
    max_depth: int = 100,
) -> dict:
    """Apply an integer operation to corresponding values in two dictionaries.

    Recursively combines two dictionaries by applying the given operation to integer
    values at matching keys.

    Supports nested dictionaries.

    Args:
        left: First dictionary to combine.
        right: Second dictionary to combine.
        op: Binary operation function to apply to integer values.
        default: Default value to use when a key is missing from a dictionary.
        depth: Current recursion depth (used internally).
        max_depth: Maximum recursion depth (to prevent infinite loops).

    Returns:
        A new dictionary with combined values.

    Raises:
        ValueError: If `max_depth` is exceeded or if value types are not supported.
    """

    def _is_numeric(x: Any) -> bool:
        """Check if x is int or float (but not bool)."""
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if depth >= max_depth:
        msg = f"{max_depth=} exceeded, unable to combine dicts."
        raise ValueError(msg)
    combined: dict = {}
    for k in set(left).union(right):
        left_val = left.get(k, default)
        right_val = right.get(k, default)
        if _is_numeric(left_val) and _is_numeric(right_val):
            # Convert floats to ints for the operation
            combined[k] = op(int(left_val), int(right_val))
        elif isinstance(left_val, dict) and isinstance(right_val, dict):
            combined[k] = _dict_int_op(
                left_val,
                right_val,
                op,
                default=default,
                depth=depth + 1,
                max_depth=max_depth,
            )
        else:
            types = [type(d[k]) for d in (left, right) if k in d]
            msg = (
                f"Unknown value types: {types}. Only dict and numeric (int/float) "
                f"values are supported."
            )
            raise ValueError(msg)  # noqa: TRY004
    return combined
