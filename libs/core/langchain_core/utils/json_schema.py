"""Utilities for JSON Schema."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from collections.abc import Sequence


def _retrieve_ref(path: str, schema: dict) -> Union[list, dict]:
    components = path.split("/")
    if components[0] != "#":
        msg = (
            "ref paths are expected to be URI fragments, meaning they should start "
            "with #."
        )
        raise ValueError(msg)
    out: Union[list, dict] = schema
    for component in components[1:]:
        if component in out:
            if isinstance(out, list):
                msg = f"Reference '{path}' not found."
                raise KeyError(msg)
            out = out[component]
        elif component.isdigit():
            index = int(component)
            if (isinstance(out, list) and 0 <= index < len(out)) or (
                isinstance(out, dict) and index in out
            ):
                out = out[index]
            else:
                msg = f"Reference '{path}' not found."
                raise KeyError(msg)
        else:
            msg = f"Reference '{path}' not found."
            raise KeyError(msg)
    return deepcopy(out)


def _dereference_refs_helper(
    obj: Any,
    full_schema: dict[str, Any],
    processed_refs: Optional[set[str]],
    skip_keys: Sequence[str],
    shallow_refs: bool,  # noqa: FBT001
) -> Any:
    """Inline every pure {'$ref':...}.

    But:

    - if shallow_refs=True: only break cycles, do not inline nested refs
    - if shallow_refs=False: deep-inline all nested refs

    Also skip recursion under any key in skip_keys.

    Returns:
        The object with refs dereferenced.
    """
    if processed_refs is None:
        processed_refs = set()

    # Handle $ref nodes (both pure and mixed)
    if isinstance(obj, dict) and "$ref" in obj:
        ref_path = obj["$ref"]
        other_props = {k: v for k, v in obj.items() if k != "$ref"}
        
        # Handle cycles
        if ref_path in processed_refs:
            # Return non-ref properties to avoid infinite recursion
            return _dereference_refs_helper(
                other_props, full_schema, processed_refs, skip_keys, shallow_refs
            )

        processed_refs.add(ref_path)

        # Resolve the reference
        target = deepcopy(_retrieve_ref(ref_path, full_schema))
        resolved_ref = _dereference_refs_helper(
            target, full_schema, processed_refs, skip_keys, shallow_refs
        )

        # If there are no other properties, return the resolved reference directly
        if not other_props:
            processed_refs.remove(ref_path)
            return resolved_ref

        # Merge resolved reference with other properties
        result_dict = {}
        if isinstance(resolved_ref, dict):
            result_dict.update(resolved_ref)

        # Process and add other properties
        for k, v in other_props.items():
            if k in skip_keys:
                result_dict[k] = deepcopy(v)
            elif isinstance(v, (dict, list)):
                result_dict[k] = _dereference_refs_helper(
                    v, full_schema, processed_refs, skip_keys, shallow_refs
                )
            else:
                result_dict[k] = v

        processed_refs.remove(ref_path)
        return result_dict

    # No $ref: recurse, skipping any keys in skip_keys
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in skip_keys:
                # do not recurse under this key
                out[k] = deepcopy(v)
            elif isinstance(v, (dict, list)):
                out[k] = _dereference_refs_helper(
                    v, full_schema, processed_refs, skip_keys, shallow_refs
                )
            else:
                out[k] = v
        return out

    if isinstance(obj, list):
        return [
            _dereference_refs_helper(
                item, full_schema, processed_refs, skip_keys, shallow_refs
            )
            for item in obj
        ]

    return obj


def dereference_refs(
    schema_obj: dict,
    *,
    full_schema: Optional[dict] = None,
    skip_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Try to substitute $refs in JSON Schema.

    Args:
      schema_obj: The fragment to dereference.
      full_schema: The complete schema (defaults to schema_obj).
      skip_keys:
        - If None (the default), we skip recursion under '$defs' *and* only
          shallow-inline refs.
        - If provided (even as an empty list), we will recurse under every key and
          deep-inline all refs.

    Returns:
        The schema with refs dereferenced.
    """
    full = full_schema or schema_obj
    keys_to_skip = list(skip_keys) if skip_keys is not None else ["$defs"]
    shallow = skip_keys is None
    return _dereference_refs_helper(schema_obj, full, None, keys_to_skip, shallow)
