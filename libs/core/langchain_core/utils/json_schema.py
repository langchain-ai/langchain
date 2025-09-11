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


def _process_dict_properties(
    properties: dict[str, Any],
    full_schema: dict[str, Any],
    processed_refs: set[str],
    skip_keys: Sequence[str],
    shallow_refs: bool,
) -> dict[str, Any]:
    """Process dictionary properties, recursing into nested structures."""
    result: dict[str, Any] = {}
    for k, v in properties.items():
        if k in skip_keys:
            result[k] = deepcopy(v)
        elif isinstance(v, (dict, list)):
            result[k] = _dereference_refs_helper(
                v, full_schema, processed_refs, skip_keys, shallow_refs
            )
        else:
            result[k] = v
    return result


def _dereference_refs_helper(
    obj: Any,
    full_schema: dict[str, Any],
    processed_refs: Optional[set[str]],
    skip_keys: Sequence[str],
    shallow_refs: bool,  # noqa: FBT001
) -> Any:
    """Dereference JSON Schema $ref objects, handling both pure and mixed references.

    This function processes JSON Schema objects containing $ref properties by resolving
    the references and merging any additional properties. It handles:

    - Pure $ref objects: {"$ref": "#/path/to/definition"}
    - Mixed $ref objects: {"$ref": "#/path", "title": "Custom Title", ...}
    - Circular references by breaking cycles and preserving non-ref properties

    Args:
        obj: The object to process (can be dict, list, or primitive)
        full_schema: The complete schema containing all definitions
        processed_refs: Set tracking currently processing refs (for cycle detection)
        skip_keys: Keys under which to skip recursion
        shallow_refs: If True, only break cycles; if False, deep-inline all refs

    Returns:
        The object with $ref properties resolved and merged with other properties.
    """
    if processed_refs is None:
        processed_refs = set()

    # Handle $ref nodes (both pure and mixed)
    if isinstance(obj, dict) and "$ref" in obj:
        ref_path = obj["$ref"]
        other_props = {k: v for k, v in obj.items() if k != "$ref"}

        # Handle cycles: return only non-ref properties to avoid infinite recursion
        if ref_path in processed_refs:
            return _process_dict_properties(
                other_props, full_schema, processed_refs, skip_keys, shallow_refs
            )

        processed_refs.add(ref_path)

        # Resolve the reference
        target = deepcopy(_retrieve_ref(ref_path, full_schema))
        resolved_ref = _dereference_refs_helper(
            target, full_schema, processed_refs, skip_keys, shallow_refs
        )

        # Pure $ref case: return resolved reference directly
        if not other_props:
            processed_refs.remove(ref_path)
            return resolved_ref

        # Mixed $ref case: merge resolved reference with other properties
        result_dict = {}
        if isinstance(resolved_ref, dict):
            result_dict.update(resolved_ref)

        # Process and merge other properties
        processed_other_props = _process_dict_properties(
            other_props, full_schema, processed_refs, skip_keys, shallow_refs
        )
        result_dict.update(processed_other_props)

        processed_refs.remove(ref_path)
        return result_dict

    # Handle regular dictionaries
    if isinstance(obj, dict):
        return _process_dict_properties(
            obj, full_schema, processed_refs, skip_keys, shallow_refs
        )

    # Handle lists
    if isinstance(obj, list):
        return [
            _dereference_refs_helper(
                item, full_schema, processed_refs, skip_keys, shallow_refs
            )
            for item in obj
        ]

    # Return primitives as-is
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
