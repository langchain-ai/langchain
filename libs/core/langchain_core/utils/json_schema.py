"""Utilities for JSON Schema."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _retrieve_ref(path: str, schema: dict) -> list | dict:
    components = path.split("/")
    if components[0] != "#":
        msg = (
            "ref paths are expected to be URI fragments, meaning they should start "
            "with #."
        )
        raise ValueError(msg)
    out: list | dict = schema
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
    *,
    shallow_refs: bool,
) -> dict[str, Any]:
    """Process dictionary properties, recursing into nested structures."""
    result: dict[str, Any] = {}
    for key, value in properties.items():
        if key in skip_keys:
            # Skip recursion for specified keys, just copy the value as-is
            result[key] = deepcopy(value)
        elif isinstance(value, (dict, list)):
            # Recursively process nested objects and arrays
            result[key] = _dereference_refs_helper(
                value, full_schema, processed_refs, skip_keys, shallow_refs
            )
        else:
            # Copy primitive values directly
            result[key] = value
    return result


def _dereference_refs_helper(
    obj: Any,
    full_schema: dict[str, Any],
    processed_refs: set[str] | None,
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
        shallow_refs: If `True`, only break cycles; if False, deep-inline all refs

    Returns:
        The object with $ref properties resolved and merged with other properties.
    """
    if processed_refs is None:
        processed_refs = set()

    # Case 1: Object contains a $ref property (pure or mixed with additional properties)
    if isinstance(obj, dict) and "$ref" in obj:
        ref_path = obj["$ref"]
        additional_properties = {
            key: value for key, value in obj.items() if key != "$ref"
        }

        # Detect circular reference: if we're already processing this $ref,
        # return only the additional properties to break the cycle
        if ref_path in processed_refs:
            return _process_dict_properties(
                additional_properties,
                full_schema,
                processed_refs,
                skip_keys,
                shallow_refs=shallow_refs,
            )

        # Mark this reference as being processed (for cycle detection)
        processed_refs.add(ref_path)

        # Fetch and recursively resolve the referenced object
        referenced_object = deepcopy(_retrieve_ref(ref_path, full_schema))
        resolved_reference = _dereference_refs_helper(
            referenced_object, full_schema, processed_refs, skip_keys, shallow_refs
        )

        # Clean up: remove from processing set before returning
        processed_refs.remove(ref_path)

        # Pure $ref case: no additional properties, return resolved reference directly
        if not additional_properties:
            return resolved_reference

        # Mixed $ref case: merge resolved reference with additional properties
        # Additional properties take precedence over resolved properties
        merged_result = {}
        if isinstance(resolved_reference, dict):
            merged_result.update(resolved_reference)

        # Process additional properties and merge them (they override resolved ones)
        processed_additional = _process_dict_properties(
            additional_properties,
            full_schema,
            processed_refs,
            skip_keys,
            shallow_refs=shallow_refs,
        )
        merged_result.update(processed_additional)

        return merged_result

    # Case 2: Regular dictionary without $ref - process all properties
    if isinstance(obj, dict):
        return _process_dict_properties(
            obj, full_schema, processed_refs, skip_keys, shallow_refs=shallow_refs
        )

    # Case 3: List - recursively process each item
    if isinstance(obj, list):
        return [
            _dereference_refs_helper(
                item, full_schema, processed_refs, skip_keys, shallow_refs
            )
            for item in obj
        ]

    # Case 4: Primitive value (string, number, boolean, null) - return unchanged
    return obj


def dereference_refs(
    schema_obj: dict,
    *,
    full_schema: dict | None = None,
    skip_keys: Sequence[str] | None = None,
) -> dict:
    """Resolve and inline JSON Schema $ref references in a schema object.

    This function processes a JSON Schema and resolves all $ref references by replacing
    them with the actual referenced content. It handles both simple references and
    complex cases like circular references and mixed $ref objects that contain
    additional properties alongside the $ref.

    Args:
        schema_obj: The JSON Schema object or fragment to process. This can be a
            complete schema or just a portion of one.
        full_schema: The complete schema containing all definitions that $refs might
            point to. If not provided, defaults to schema_obj (useful when the
            schema is self-contained).
        skip_keys: Controls recursion behavior and reference resolution depth:
            - If `None` (Default): Only recurse under '$defs' and use shallow reference
              resolution (break cycles but don't deep-inline nested refs)
            - If provided (even as []): Recurse under all keys and use deep reference
              resolution (fully inline all nested references)

    Returns:
        A new dictionary with all $ref references resolved and inlined. The original
        schema_obj is not modified.

    Examples:
        Basic reference resolution:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {"name": {"$ref": "#/$defs/string_type"}},
        ...     "$defs": {"string_type": {"type": "string"}},
        ... }
        >>> result = dereference_refs(schema)
        >>> result["properties"]["name"]  # {"type": "string"}

        Mixed $ref with additional properties:
        >>> schema = {
        ...     "properties": {
        ...         "name": {"$ref": "#/$defs/base", "description": "User name"}
        ...     },
        ...     "$defs": {"base": {"type": "string", "minLength": 1}},
        ... }
        >>> result = dereference_refs(schema)
        >>> result["properties"]["name"]
        # {"type": "string", "minLength": 1, "description": "User name"}

        Handling circular references:
        >>> schema = {
        ...     "properties": {"user": {"$ref": "#/$defs/User"}},
        ...     "$defs": {
        ...         "User": {
        ...             "type": "object",
        ...             "properties": {"friend": {"$ref": "#/$defs/User"}},
        ...         }
        ...     },
        ... }
        >>> result = dereference_refs(schema)  # Won't cause infinite recursion

    Note:
        - Circular references are handled gracefully by breaking cycles
        - Mixed $ref objects (with both $ref and other properties) are supported
        - Additional properties in mixed $refs override resolved properties
        - The $defs section is preserved in the output by default
    """
    full = full_schema or schema_obj
    keys_to_skip = list(skip_keys) if skip_keys is not None else ["$defs"]
    shallow = skip_keys is None
    return _dereference_refs_helper(schema_obj, full, None, keys_to_skip, shallow)
