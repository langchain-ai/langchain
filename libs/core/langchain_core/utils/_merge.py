from __future__ import annotations

from typing import Any, Dict, List, Optional


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dicts, handling specific scenarios where a key exists in both
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
    for right_k, right_v in right.items():
        if right_k not in merged:
            merged[right_k] = right_v
        elif right_v is not None and merged[right_k] is None:
            merged[right_k] = right_v
        elif right_v is None:
            continue
        elif type(merged[right_k]) != type(right_v):
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


def merge_lists(left: Optional[List], right: Optional[List]) -> Optional[List]:
    """Add two lists, handling None."""
    if left is None and right is None:
        return None
    elif left is None or right is None:
        return left or right
    else:
        merged = left.copy()
        for e in right:
            if isinstance(e, dict) and "index" in e and isinstance(e["index"], int):
                to_merge = [
                    i
                    for i, e_left in enumerate(merged)
                    if e_left["index"] == e["index"]
                ]
                if to_merge:
                    merged[to_merge[0]] = merge_dicts(merged[to_merge[0]], e)
                else:
                    merged = merged + [e]
            else:
                merged = merged + [e]
        return merged
