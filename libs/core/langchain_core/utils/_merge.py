from __future__ import annotations

from typing import Any, Dict, List, Optional


def merge_dicts(left: Dict[str, Any], *others: Dict[str, Any]) -> Dict[str, Any]:
    """Merge many dicts, handling specific scenarios where a key exists in both
    dictionaries but has a value of None in 'left'. In such cases, the method uses the
    value from 'right' for that key in the merged dictionary.

    Example:
        If left = {"function_call": {"arguments": None}} and
        right = {"function_call": {"arguments": "{\n"}}
        then, after merging, for the key "function_call",
        the value from 'right' is used,
        resulting in merged = {"function_call": {"arguments": "{\n"}}.
    """
    merged = left.copy()
    for right in others:
        for right_k, right_v in right.items():
            if right_k not in merged:
                merged[right_k] = right_v
            elif right_v is not None and merged[right_k] is None:
                merged[right_k] = right_v
            elif right_v is None:
                continue
            elif type(merged[right_k]) is not type(right_v):
                raise TypeError(
                    f'additional_kwargs["{right_k}"] already exists in this message,'
                    " but with a different type."
                )
            elif isinstance(merged[right_k], str):
                merged[right_k] += right_v
            elif isinstance(merged[right_k], dict):
                merged[right_k] = merge_dicts(merged[right_k], right_v)
            elif isinstance(merged[right_k], list):
                merged[right_k] = merge_lists(merged[right_k], right_v)
            elif merged[right_k] == right_v:
                continue
            else:
                raise TypeError(
                    f"Additional kwargs key {right_k} already exists in left dict and "
                    f"value has unsupported type {type(merged[right_k])}."
                )
    return merged


def merge_lists(left: Optional[List], *others: Optional[List]) -> Optional[List]:
    """Add many lists, handling None."""
    merged = left.copy() if left is not None else None
    for other in others:
        if other is None:
            continue
        elif merged is None:
            merged = other.copy()
        else:
            for e in other:
                if isinstance(e, dict) and "index" in e and isinstance(e["index"], int):
                    to_merge = [
                        i
                        for i, e_left in enumerate(merged)
                        if e_left["index"] == e["index"]
                    ]
                    if to_merge:
                        # If a top-level "type" has been set for a chunk, it should no
                        # longer be overridden by the "type" field in future chunks.
                        if "type" in merged[to_merge[0]] and "type" in e:
                            e.pop("type")
                        merged[to_merge[0]] = merge_dicts(merged[to_merge[0]], e)
                    else:
                        merged.append(e)
                else:
                    merged.append(e)
    return merged


def merge_obj(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return left if left is not None else right
    elif type(left) is not type(right):
        raise TypeError(
            f"left and right are of different types. Left type:  {type(left)}. Right "
            f"type: {type(right)}."
        )
    elif isinstance(left, str):
        return left + right
    elif isinstance(left, dict):
        return merge_dicts(left, right)
    elif isinstance(left, list):
        return merge_lists(left, right)
    elif left == right:
        return left
    else:
        raise ValueError(
            f"Unable to merge {left=} and {right=}. Both must be of type str, dict, or "
            f"list, or else be two equal objects."
        )
