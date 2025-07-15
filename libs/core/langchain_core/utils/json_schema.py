"""Utilities for JSON Schema."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from collections.abc import Sequence

def _retrieve_ref(path: str, schema: dict) -> dict:
    """Return the schema fragment pointed to by an internal ``$ref``.

    This supports the subset of JSON-Pointer used by JSON-Schema where every
    reference **must** be a URI fragment (i.e. it starts with ``#``).  Each
    “/”-separated token identifies either:

    * a **mapping key** when the current node is a ``dict``; or
    * a **zero-based list index** when the current node is a ``list``.

    Parameters
    ----------
    path :
        The reference exactly as it appears in the schema
        (e.g. ``"#/properties/name"``). **Must** start with ``#``.
    schema :
        The document-root schema object inside which *path* is resolved.
        This object is **never** mutated.

    Returns
    -------
    dict
        A **deep copy** of the schema fragment located at *path*.

    Raises
    ------
    ValueError
        If *path* does **not** start with ``#``.
    KeyError
        If any token cannot be resolved.
    """
    tokens = path.split("/")

    # All internal JSON-Schema references must be URI fragments.
    if tokens[0] != "#":
        raise ValueError(
            "ref paths are expected to be URI fragments, meaning they should "
            "start with '#'.",
        )

    node: Any = schema  # start at the document root

    for token in tokens[1:]:
        # ----- Mapping lookup -------------------------------------------------- #
        if isinstance(node, dict):
            if token in node:
                node = node[token]
                continue
            # Numeric token may reference an int key stored in the mapping.
            if token.isdigit() and (int_token := int(token)) in node:
                node = node[int_token]
                continue

        # ----- Sequence index -------------------------------------------------- #
        if token.isdigit() and isinstance(node, list):
            idx = int(token)
            if idx >= len(node):
                msg = f"Index {idx} out of range while resolving {path!r}"
                raise KeyError(msg)
            node = node[idx]
            continue

        # ---------------------------------------------------------------------- #
        msg = f"Unable to resolve token {token!r} in {path!r}"
        raise KeyError(msg)

    # Hand back a deep copy so callers can mutate safely.
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
