"""Utilities for JSON Schema."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from collections.abc import Sequence


def _retrieve_ref(path: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve an internal JSON-Schema “$ref” (JSON Pointer) and return the
    referenced schema fragment **as a deep copy**.

    $ref pointers in JSON Schema follow the same rules as RFC 6901 JSON Pointer:
        https://datatracker.ietf.org/doc/html/rfc6901

    The function supports the subset of JSON Pointer used by JSON-Schema:
    every reference is a URI-fragment (it begins with “#”) and each path
    segment, separated by “/”, identifies either

    * a **mapping key** when the current node is a ``dict``; or  
    * a **zero-based list index** when the current node is a ``list``.

    Examples
    --------
    ``#/properties/address``               – selects a dict key  
    ``#/items/2/properties/name``          – selects a list index then a key  

    Parameters
    ----------
    path
        The ``$ref`` value exactly as written in the schema (e.g.
        ``"#/properties/foo/anyOf/1"``).  Must start with “#”.
    schema
        The root schema object inside which the reference should be resolved.
        It is **never mutated**; the return value is a detached copy.

    Returns
    -------
    dict
        The schema fragment located at *path*, copied deeply so callers may
        modify it without affecting the original document.

    Raises
    ------
    ValueError
        If *path* does **not** start with “#”.
    KeyError
        If any segment cannot be resolved―either a mapping key is missing or
        a list index is out of range.
    """
    # Break the URI fragment into its individual tokens
    tokens = path.split("/")

    # Per JSON-Schema spec, internal references must be URI fragments
    if tokens[0] != "#":
        raise ValueError("Reference paths must be URI fragments starting with '#'.")

    node: Any = schema  # begin at the document root

    # Walk the tokens one by one, drilling down the schema structure
    for token in tokens[1:]:
        if isinstance(node, dict) and token in node:
            # -- Mapping lookup --------------------------------------------------
            node = node[token]
        elif token.isdigit() and isinstance(node, list):
            # -- Sequence index --------------------------------------------------
            idx = int(token)
            if idx >= len(node):
                raise KeyError(f"Index {idx} out of range while resolving {path!r}")
            node = node[idx]
        else:
            # -- Neither dict key nor list index matched -------------------------
            raise KeyError(f"Unable to resolve token {token!r} in {path!r}")

    # Hand back a deep copy so callers can tweak the fragment safely
    return deepcopy(node)


def _dereference_refs_helper(
    obj: Any,
    full_schema: dict[str, Any],
    skip_keys: Sequence[str],
    processed_refs: Optional[set[str]] = None,
) -> Any:
    if processed_refs is None:
        processed_refs = set()

    if isinstance(obj, dict):
        obj_out = {}
        for k, v in obj.items():
            if k in skip_keys:
                obj_out[k] = v
            elif k == "$ref":
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                full_ref = _dereference_refs_helper(
                    ref, full_schema, skip_keys, processed_refs
                )
                processed_refs.remove(v)
                return full_ref
            elif isinstance(v, (list, dict)):
                obj_out[k] = _dereference_refs_helper(
                    v, full_schema, skip_keys, processed_refs
                )
            else:
                obj_out[k] = v
        return obj_out
    if isinstance(obj, list):
        return [
            _dereference_refs_helper(el, full_schema, skip_keys, processed_refs)
            for el in obj
        ]
    return obj


def _infer_skip_keys(
    obj: Any, full_schema: dict, processed_refs: Optional[set[str]] = None
) -> list[str]:
    if processed_refs is None:
        processed_refs = set()

    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$ref":
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                keys.append(v.split("/")[1])
                keys += _infer_skip_keys(ref, full_schema, processed_refs)
            elif isinstance(v, (list, dict)):
                keys += _infer_skip_keys(v, full_schema, processed_refs)
    elif isinstance(obj, list):
        for el in obj:
            keys += _infer_skip_keys(el, full_schema, processed_refs)
    return keys


def dereference_refs(
    schema_obj: dict,
    *,
    full_schema: Optional[dict] = None,
    skip_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Try to substitute $refs in JSON Schema.

    Args:
        schema_obj: The schema object to dereference.
        full_schema: The full schema object. Defaults to None.
        skip_keys: The keys to skip. Defaults to None.

    Returns:
        The dereferenced schema object.
    """
    full_schema = full_schema or schema_obj
    skip_keys = (
        skip_keys
        if skip_keys is not None
        else _infer_skip_keys(schema_obj, full_schema)
    )
    return _dereference_refs_helper(schema_obj, full_schema, skip_keys)
