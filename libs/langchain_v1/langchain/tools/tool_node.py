"""Tool node with enhanced state injection support.

This module provides a ``ToolNode`` subclass that adds safe handling for
``NotRequired`` (and ``Optional``) fields when using ``InjectedState``.
The upstream ``langgraph`` ``ToolNode`` raises a ``KeyError`` (or
``AttributeError``) when an ``InjectedState("<field>")`` annotation
references a state field declared as ``NotRequired`` and that field is
absent from the current state.  The subclass defined here overrides
``_inject_tool_args`` to use ``.get()`` with sensible defaults so that
missing ``NotRequired`` fields resolve to ``None`` instead of raising.
"""

from __future__ import annotations

from copy import copy
from typing import Any, get_args, get_origin, get_type_hints

from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState, InjectedStore, ToolRuntime
from langgraph.prebuilt.tool_node import (
    ToolCallRequest,
    ToolCallWithContext,
    ToolCallWrapper,
    ToolNode as _ToolNode,
    _get_all_injected_args,
)
from typing_extensions import Annotated, NotRequired, Required

from langchain_core.messages import ToolCall

__all__ = [
    "InjectedState",
    "InjectedStore",
    "ToolCallRequest",
    "ToolCallWithContext",
    "ToolCallWrapper",
    "ToolNode",
    "ToolRuntime",
]


def _is_not_required(annotation: Any) -> bool:
    """Return ``True`` if *annotation* wraps ``NotRequired[...]``.

    Handles raw ``NotRequired[T]`` as well as
    ``NotRequired[Annotated[T, ...]]`` forms that ``typing_extensions``
    emits.

    Args:
        annotation: A type annotation to inspect.

    Returns:
        ``True`` when *annotation* is ``NotRequired[...]``.
    """
    origin = get_origin(annotation)
    if origin is NotRequired:
        return True
    return False


def _is_not_required_string(annotation: Any) -> bool:
    """Return ``True`` if *annotation* is a stringified ``NotRequired[...]``.

    When ``from __future__ import annotations`` is active, all
    annotations are stored as strings (or ``ForwardRef`` objects).
    This helper detects that case by inspecting the string
    representation.

    Args:
        annotation: A type annotation to inspect (may be a string or
            ``ForwardRef``).

    Returns:
        ``True`` when the annotation textually wraps ``NotRequired``.
    """
    text: str | None = None
    if isinstance(annotation, str):
        text = annotation
    else:
        # typing.ForwardRef stores the original string in __arg__
        arg = getattr(annotation, "__arg__", None)
        if isinstance(arg, str):
            text = arg
    if text is not None:
        return text.startswith("NotRequired[")
    return False


def _get_not_required_fields(state_schema: type | None) -> set[str]:
    """Collect the names of all ``NotRequired`` fields in a TypedDict schema.

    The function inspects the ``__annotations__`` (and
    ``__required_keys__`` / ``__optional_keys__`` when available) of the
    given TypedDict class to determine which fields are *not* required.

    For ``total=True`` TypedDicts (the default), a field is not-required
    only when explicitly wrapped in ``NotRequired[...]``.  For
    ``total=False`` TypedDicts, every field that is *not* wrapped in
    ``Required[...]`` is considered not-required.

    Args:
        state_schema: A ``TypedDict`` subclass used as the graph state
            schema, or ``None``.

    Returns:
        A set of field names that are not-required.  Returns an empty set
        when *state_schema* is ``None`` or is not a TypedDict.
    """
    if state_schema is None:
        return set()

    # typing_extensions TypedDicts expose __optional_keys__.  However,
    # when ``from __future__ import annotations`` is active the
    # TypedDict metaclass cannot resolve ``NotRequired[...]`` wrappers
    # and ``__optional_keys__`` may be an empty frozenset even though
    # some fields *are* optional.  We therefore treat a non-empty
    # ``__optional_keys__`` as authoritative but fall through to
    # annotation inspection otherwise.
    optional_keys: frozenset[str] | None = getattr(
        state_schema, "__optional_keys__", None
    )
    if optional_keys:
        return set(optional_keys)

    # Fallback: resolve annotations via get_type_hints (handles forward
    # references from ``from __future__ import annotations``) and
    # inspect them for NotRequired wrappers.
    try:
        annotations = get_type_hints(state_schema, include_extras=True)
    except Exception:
        # get_type_hints can fail on complex schemas with unresolvable
        # forward references.  Fall back to raw __annotations__.
        annotations = getattr(state_schema, "__annotations__", {})

    not_required: set[str] = set()
    total: bool = getattr(state_schema, "__total__", True)
    for field_name, field_type in annotations.items():
        if _is_not_required(field_type):
            not_required.add(field_name)
        elif _is_not_required_string(field_type):
            # Handle stringified annotations from
            # ``from __future__ import annotations``
            not_required.add(field_name)
        elif not total and get_origin(field_type) is not Required:
            # In a total=False TypedDict, fields not marked Required are
            # optional.
            not_required.add(field_name)
    return not_required


class ToolNode(_ToolNode):
    """``ToolNode`` with safe handling for ``NotRequired`` state fields.

    When a tool parameter is annotated with ``InjectedState("<field>")``,
    the upstream ``_ToolNode._inject_tool_args`` accesses the state via
    direct subscript (``state[field]``), which raises ``KeyError`` if the
    field is absent.  This subclass overrides the injection logic so that
    fields declared as ``NotRequired`` in the state schema are accessed
    via ``.get(field)`` instead, returning ``None`` when the field has not
    been populated.

    The ``state_schema`` is an *optional* parameter.  When not supplied
    the ``ToolNode`` still works but falls back to ``.get()`` for **all**
    field-level ``InjectedState`` accesses on ``dict`` states, which is a
    safe superset of the original behaviour (``state.get(k)`` returns the
    same value as ``state[k]`` when the key exists).
    """

    def __init__(
        self,
        *args: Any,
        state_schema: type | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ``ToolNode``.

        Args:
            *args: Positional arguments forwarded to the parent
                ``ToolNode.__init__``.
            state_schema: An optional ``TypedDict`` class used as the
                graph state schema.  When provided the node can
                distinguish ``NotRequired`` fields from required ones and
                only relax access for the former.
            **kwargs: Keyword arguments forwarded to the parent
                ``ToolNode.__init__``.
        """
        super().__init__(*args, **kwargs)
        self._state_schema = state_schema
        self._not_required_fields: set[str] = _get_not_required_fields(
            state_schema
        )

    def _inject_tool_args(
        self,
        tool_call: ToolCall,
        tool_runtime: ToolRuntime,
        tool: BaseTool | None = None,
    ) -> ToolCall:
        """Inject graph state, store, and runtime into tool call arguments.

        This override adds safe handling for ``NotRequired`` state fields.
        When a state field referenced by ``InjectedState("<field>")`` is
        declared as ``NotRequired`` in the state schema (or when no schema
        is known), the field is accessed via ``.get()`` so that missing
        keys resolve to ``None`` instead of raising ``KeyError``.

        Args:
            tool_call: The tool call dictionary to augment.
            tool_runtime: The ``ToolRuntime`` with state, config, store,
                context, and stream_writer.
            tool: Optional tool instance for dynamically registered tools.

        Returns:
            A new ``ToolCall`` dict with injected arguments.

        Raises:
            ValueError: If store injection is required but no store is
                provided, or if state injection requirements cannot be
                satisfied.
        """
        injected = self._injected_args.get(tool_call["name"])
        if not injected and tool is not None:
            injected = _get_all_injected_args(tool)
        if not injected:
            return tool_call

        tool_call_copy: ToolCall = copy(tool_call)
        injected_args: dict[str, Any] = {}

        # --- Inject state (with NotRequired awareness) ---
        if injected.state:
            state = tool_runtime.state
            # Handle list state by converting to dict
            if isinstance(state, list):
                required_fields = list(injected.state.values())
                if (
                    len(required_fields) == 1
                    and required_fields[0] == self._messages_key
                ) or required_fields[0] is None:
                    state = {self._messages_key: state}
                else:
                    err_msg = (
                        f"Invalid input to ToolNode. Tool {tool_call['name']} "
                        f"requires graph state dict as input."
                    )
                    if any(
                        state_field
                        for state_field in injected.state.values()
                    ):
                        required_fields_str = ", ".join(
                            f for f in required_fields if f
                        )
                        err_msg += (
                            f" State should contain fields "
                            f"{required_fields_str}."
                        )
                    raise ValueError(err_msg)

            # Extract state values with NotRequired safety
            if isinstance(state, dict):
                for tool_arg, state_field in injected.state.items():
                    if state_field is None:
                        # Entire state requested
                        injected_args[tool_arg] = state
                    elif self._is_field_not_required(state_field):
                        # Field is NotRequired or schema is unknown:
                        # use .get() so missing keys yield None
                        injected_args[tool_arg] = state.get(state_field)
                    else:
                        # Field is required: preserve original behaviour
                        injected_args[tool_arg] = state[state_field]
            else:
                for tool_arg, state_field in injected.state.items():
                    if state_field is None:
                        injected_args[tool_arg] = state
                    elif self._is_field_not_required(state_field):
                        injected_args[tool_arg] = getattr(
                            state, state_field, None
                        )
                    else:
                        injected_args[tool_arg] = getattr(
                            state, state_field
                        )

        # --- Inject store ---
        if injected.store:
            if tool_runtime.store is None:
                msg = (
                    "Cannot inject store into tools with InjectedStore "
                    "annotations - please compile your graph with a store."
                )
                raise ValueError(msg)
            injected_args[injected.store] = tool_runtime.store

        # --- Inject runtime ---
        if injected.runtime:
            injected_args[injected.runtime] = tool_runtime

        tool_call_copy["args"] = {**tool_call_copy["args"], **injected_args}
        return tool_call_copy

    def _is_field_not_required(self, field_name: str) -> bool:
        """Check whether *field_name* is a ``NotRequired`` state field.

        When no ``state_schema`` was provided (i.e. we have no schema
        information), we conservatively treat **all** fields as
        potentially not-required and use safe ``.get()`` access.  This
        avoids ``KeyError`` at the cost of returning ``None`` for truly
        missing required fields (which would have raised anyway).

        Args:
            field_name: The state field name to check.

        Returns:
            ``True`` if the field should be accessed via ``.get()``.
        """
        if self._state_schema is None:
            # No schema known: be safe and use .get() for all fields
            return True
        return field_name in self._not_required_fields
