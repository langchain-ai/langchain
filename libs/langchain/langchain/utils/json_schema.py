from __future__ import annotations

from typing import Optional, TypeVar, Union, cast


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
    return out


JSON_LIKE = TypeVar("JSON_LIKE", bound=Union[dict, list])


def _dereference_refs_helper(obj: JSON_LIKE, full_schema: dict) -> JSON_LIKE:
    if isinstance(obj, dict):
        obj_out = {}
        for k, v in obj.items():
            if k == "$ref":
                ref = _retrieve_ref(v, full_schema)
                obj_out[k] = _dereference_refs_helper(ref, full_schema)
            elif isinstance(v, (list, dict)):
                obj_out[k] = _dereference_refs_helper(v, full_schema)  # type: ignore
            else:
                obj_out[k] = v
        return cast(JSON_LIKE, obj_out)
    elif isinstance(obj, list):
        return cast(
            JSON_LIKE, [_dereference_refs_helper(el, full_schema) for el in obj]
        )
    else:
        return obj


def dereference_refs(
    schema_obj: dict, *, full_schema: Optional[dict] = None
) -> Union[dict, list]:
    """Try to substitute $refs in JSON Schema."""

    full_schema = full_schema or schema_obj
    return _dereference_refs_helper(schema_obj, full_schema)
