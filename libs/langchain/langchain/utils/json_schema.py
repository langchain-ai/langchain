from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, Sequence


def _retrieve_ref(path: str, schema: dict) -> dict:
    components = path.split("/")
    if components[0] != "#":
        raise ValueError(
            "ref paths are expected to be URI fragments, meaning they should start "
            "with #."
        )
    out = schema
    for component in components[1:]:
        out = out[component]
    return deepcopy(out)


def _dereference_refs_helper(
    obj: Any, full_schema: dict, skip_keys: Sequence[str]
) -> Any:
    if isinstance(obj, dict):
        obj_out = {}
        for k, v in obj.items():
            if k in skip_keys:
                obj_out[k] = v
            elif k == "$ref":
                ref = _retrieve_ref(v, full_schema)
                return _dereference_refs_helper(ref, full_schema, skip_keys)
            elif isinstance(v, (list, dict)):
                obj_out[k] = _dereference_refs_helper(v, full_schema, skip_keys)
            else:
                obj_out[k] = v
        return obj_out
    elif isinstance(obj, list):
        return [_dereference_refs_helper(el, full_schema, skip_keys) for el in obj]
    else:
        return obj


def _infer_skip_keys(obj: Any, full_schema: dict) -> List[str]:
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$ref":
                ref = _retrieve_ref(v, full_schema)
                keys.append(v.split("/")[1])
                keys += _infer_skip_keys(ref, full_schema)
            elif isinstance(v, (list, dict)):
                keys += _infer_skip_keys(v, full_schema)
    elif isinstance(obj, list):
        for el in obj:
            keys += _infer_skip_keys(el, full_schema)
    return keys


def dereference_refs(
    schema_obj: dict,
    *,
    full_schema: Optional[dict] = None,
    skip_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Try to substitute $refs in JSON Schema."""

    full_schema = full_schema or schema_obj
    skip_keys = (
        skip_keys
        if skip_keys is not None
        else _infer_skip_keys(schema_obj, full_schema)
    )
    return _dereference_refs_helper(schema_obj, full_schema, skip_keys)
