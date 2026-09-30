"""Usage utilities."""

from collections.abc import Callable
from typing import Any


def _dict_int_op(
    left: dict[str, Any],
    right: dict[str, Any],
    op: Callable[[int, int], int],
    *,
    default: int = 0,
    depth: int = 0,
    max_depth: int = 100,
) -> dict[str, Any]:
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
    if depth >= max_depth:
        msg = f"{max_depth=} exceeded, unable to combine dicts."
        raise ValueError(msg)
    combined: dict[str, Any] = {}
    for k in set(left).union(right):
        left_val = left.get(k)
        right_val = right.get(k)

        if (isinstance(left_val, int) or left_val is None) and (
            isinstance(right_val, int) or right_val is None
        ):
            combined[k] = op(
                left_val if left_val is not None else default,
                right_val if right_val is not None else default,
            )
        elif (isinstance(left_val, dict) or left_val is None) and (
            isinstance(right_val, dict) or right_val is None
        ):
            combined[k] = _dict_int_op(
                left_val if isinstance(left_val, dict) else {},
                right_val if isinstance(right_val, dict) else {},
                op,
                default=default,
                depth=depth + 1,
                max_depth=max_depth,
            )
        else:
            types = [type(d[k]) for d in (left, right) if k in d]
            msg = (
                f"Unknown value types: {types}. Only dict and int values are supported."
            )
            raise ValueError(msg)  # noqa: TRY004
    return combined
