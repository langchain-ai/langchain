from __future__ import annotations

from typing import Any, Dict


def merge_dicts(
    left: Dict[str, Any], right: Dict[str, Any], *, merge_lists: bool = True
) -> Dict[str, Any]:
    """Merge additional_kwargs from another BaseMessageChunk into this one,
    handling specific scenarios where a key exists in both dictionaries
    but has a value of None in 'left'. In such cases, the method uses the
    value from 'right' for that key in the merged dictionary.

    Args:
        left: left dict.
        right: right dict.
        merge_lists: Whether to attempt element-wise merge of a list of dicts or
            to concatenate them.

    Example:
        If left = {"function_call": {"arguments": None}} and
        right = {"function_call": {"arguments": "{\n"}}
        then, after merging, for the key "function_call",
        the value from 'right' is used,
        resulting in merged = {"function_call": {"arguments": "{\n"}}.
    """
    merged = left.copy()
    for k, v in right.items():
        if k not in merged:
            merged[k] = v
        elif merged[k] is None and v:
            merged[k] = v
        elif v is None:
            continue
        elif merged[k] == v:
            continue
        elif type(merged[k]) != type(v):
            raise TypeError(
                f'additional_kwargs["{k}"] already exists in this message,'
                " but with a different type."
            )
        elif isinstance(merged[k], str):
            merged[k] += v
        elif isinstance(merged[k], dict):
            merged[k] = merge_dicts(merged[k], v, merge_lists=merge_lists)
        elif isinstance(merged[k], list):
            if merge_lists:
                merged[k] = merged[k].copy()
                for i, e in enumerate(v):
                    if isinstance(e, dict) and isinstance(e.get("index"), int):
                        i = e["index"]
                    if isinstance(e, dict) and i < len(merged[k]):
                        merged[k][i] = merge_dicts(
                            merged[k][i], e, merge_lists=merge_lists
                        )
                    else:
                        merged[k] = merged[k] + [e]
            else:
                merged[k] = merged[k] + v
        else:
            raise TypeError(
                f"Additional kwargs key {k} already exists in left dict and value has "
                f"unsupported type {type(merged[k])}."
            )
    return merged
