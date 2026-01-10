from __future__ import annotations

from typing import Any


def merge_dicts(left: dict[str, Any], *others: dict[str, Any]) -> dict[str, Any]:
    r"""Merge dictionaries.

    Merge many dicts, handling specific scenarios where a key exists in both
    dictionaries but has a value of `None` in `'left'`. In such cases, the method uses
    the value from `'right'` for that key in the merged dictionary.

    Args:
        left: The first dictionary to merge.
        others: The other dictionaries to merge.

    Returns:
        The merged dictionary.

    Raises:
        TypeError: If the key exists in both dictionaries but has a different type.
        TypeError: If the value has an unsupported type.

    Example:
        If `left = {"function_call": {"arguments": None}}` and
        `right = {"function_call": {"arguments": "{\n"}}`, then, after merging, for the
        key `'function_call'`, the value from `'right'` is used, resulting in
        `merged = {"function_call": {"arguments": "{\n"}}`.
    """
    merged = left.copy()
    for right in others:
        for right_k, right_v in right.items():
            if right_k not in merged or (
                right_v is not None and merged[right_k] is None
            ):
                merged[right_k] = right_v
            elif right_v is None:
                continue
            elif type(merged[right_k]) is not type(right_v):
                msg = (
                    f'additional_kwargs["{right_k}"] already exists in this message,'
                    " but with a different type."
                )
                raise TypeError(msg)
            elif isinstance(merged[right_k], str):
                # TODO: Add below special handling for 'type' key in 0.3 and remove
                # merge_lists 'type' logic.
                #
                # if right_k == "type":
                #     if merged[right_k] == right_v:
                #         continue
                #     else:
                #         raise ValueError(
                #             "Unable to merge. Two different values seen for special "
                #             f"key 'type': {merged[right_k]} and {right_v}. 'type' "
                #             "should either occur once or have the same value across "
                #             "all dicts."
                #         )
                if (right_k == "index" and merged[right_k].startswith("lc_")) or (
                    right_k in {"id", "output_version", "model_provider"}
                    and merged[right_k] == right_v
                ):
                    continue
                merged[right_k] += right_v
            elif isinstance(merged[right_k], dict):
                merged[right_k] = merge_dicts(merged[right_k], right_v)
            elif isinstance(merged[right_k], list):
                merged[right_k] = merge_lists(merged[right_k], right_v)
            elif merged[right_k] == right_v:
                continue
            elif isinstance(merged[right_k], int):
                merged[right_k] += right_v
            else:
                msg = (
                    f"Additional kwargs key {right_k} already exists in left dict and "
                    f"value has unsupported type {type(merged[right_k])}."
                )
                raise TypeError(msg)
    return merged


def merge_lists(left: list | None, *others: list | None) -> list | None:
    """Add many lists, handling `None`.

    Args:
        left: The first list to merge.
        others: The other lists to merge.

    Returns:
        The merged list.
    """
    merged = left.copy() if left is not None else None
    for other in others:
        if other is None:
            continue
        if merged is None:
            merged = other.copy()
        else:
            for e in other:
                if (
                    isinstance(e, dict)
                    and "index" in e
                    and (
                        isinstance(e["index"], int)
                        or (
                            isinstance(e["index"], str) and e["index"].startswith("lc_")
                        )
                    )
                ):
                    to_merge = [
                        i
                        for i, e_left in enumerate(merged)
                        if "index" in e_left and e_left["index"] == e["index"]
                    ]
                    if to_merge:
                        # TODO: Remove this once merge_dict is updated with special
                        # handling for 'type'.
                        if (left_type := merged[to_merge[0]].get("type")) and (
                            e.get("type") == "non_standard" and "value" in e
                        ):
                            if left_type != "non_standard":
                                # standard + non_standard
                                new_e: dict[str, Any] = {
                                    "extras": {
                                        k: v
                                        for k, v in e["value"].items()
                                        if k != "type"
                                    }
                                }
                            else:
                                # non_standard + non_standard
                                new_e = {
                                    "value": {
                                        k: v
                                        for k, v in e["value"].items()
                                        if k != "type"
                                    }
                                }
                                if "index" in e:
                                    new_e["index"] = e["index"]
                        else:
                            new_e = (
                                {k: v for k, v in e.items() if k != "type"}
                                if "type" in e
                                else e
                            )
                        merged[to_merge[0]] = merge_dicts(merged[to_merge[0]], new_e)
                    else:
                        merged.append(e)
                else:
                    merged.append(e)
    return merged


def merge_obj(left: Any, right: Any) -> Any:
    """Merge two objects.

    It handles specific scenarios where a key exists in both dictionaries but has a
    value of `None` in `'left'`. In such cases, the method uses the value from `'right'`
    for that key in the merged dictionary.

    Args:
        left: The first object to merge.
        right: The other object to merge.

    Returns:
        The merged object.

    Raises:
        TypeError: If the key exists in both dictionaries but has a different type.
        ValueError: If the two objects cannot be merged.
    """
    if left is None or right is None:
        return left if left is not None else right
    if type(left) is not type(right):
        msg = (
            f"left and right are of different types. Left type:  {type(left)}. Right "
            f"type: {type(right)}."
        )
        raise TypeError(msg)
    if isinstance(left, str):
        return left + right
    if isinstance(left, dict):
        return merge_dicts(left, right)
    if isinstance(left, list):
        return merge_lists(left, right)
    if left == right:
        return left
    msg = (
        f"Unable to merge {left=} and {right=}. Both must be of type str, dict, or "
        f"list, or else be two equal objects."
    )
    raise ValueError(msg)
