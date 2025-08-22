"""Usage utilities."""

from typing import Callable


def _dict_int_op(
    left: dict,
    right: dict,
    op: Callable[[int, int], int],
    *,
    default: int = 0,
    depth: int = 0,
    max_depth: int = 100,
) -> dict:
    if depth >= max_depth:
        msg = f"{max_depth=} exceeded, unable to combine dicts."
        raise ValueError(msg)
    combined: dict = {}
    for k in set(left).union(right):
        left_val = left.get(k, default)
        right_val = right.get(k, default)

        # Handle None values - treat as default (usually 0)
        if left_val is None:
            left_val = default
        if right_val is None:
            right_val = default

        if isinstance(left_val, int) and isinstance(right_val, int):
            combined[k] = op(left_val, right_val)
        elif isinstance(left.get(k, {}), dict) and isinstance(right.get(k, {}), dict):
            combined[k] = _dict_int_op(
                left.get(k, {}),
                right.get(k, {}),
                op,
                default=default,
                depth=depth + 1,
                max_depth=max_depth,
            )
        else:
            types = [type(d[k]) for d in (left, right) if k in d]
            msg = (
                f"Unknown value types: {types}. "
                "Only dict, int, and None values are supported."
            )
            raise ValueError(msg)  # noqa: TRY004
    return combined
