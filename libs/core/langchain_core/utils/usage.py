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
        if isinstance(left.get(k, default), int) and isinstance(
            right.get(k, default), int
        ):
            combined[k] = op(left.get(k, default), right.get(k, default))
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
                f"Unknown value types: {types}. Only dict and int values are supported."
            )
            raise ValueError(msg)  # noqa: TRY004
    return combined
