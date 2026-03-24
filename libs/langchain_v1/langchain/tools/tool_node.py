"""ToolNode with graceful handling for NotRequired state fields.

This module re-exports InjectedState-related types from langgraph and provides
a ToolNode subclass that fixes KeyError when InjectedState references a
NotRequired field that is absent from the agent state.

See: https://github.com/langchain-ai/langchain/issues/35585
"""

from __future__ import annotations

import inspect
from copy import copy
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState, InjectedStore, ToolRuntime
from langgraph.prebuilt.tool_node import (
    ToolCallRequest,
    ToolCallWithContext,
    ToolCallWrapper,
    ToolNode as _ToolNode,
)
from langgraph.prebuilt.tool_node import ToolCall

__all__ = [
    "InjectedState",
    "InjectedStore",
    "ToolCallRequest",
    "ToolCallWithContext",
    "ToolCallWrapper",
    "ToolNode",
    "ToolRuntime",
]

_SENTINEL = object()


def _tool_param_has_default(tool: BaseTool, param_name: str) -> bool:
    """Return True if the tool's underlying function has a default for *param_name*."""
    func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if func is None:
        return False
    try:
        sig = inspect.signature(func)
        p = sig.parameters.get(param_name)
        return p is not None and p.default is not inspect.Parameter.empty
    except (ValueError, TypeError):
        return False


class ToolNode(_ToolNode):
    """ToolNode that gracefully handles NotRequired InjectedState fields.

    When a tool parameter is annotated with ``InjectedState("field")`` and the
    referenced field is declared as ``NotRequired`` in the custom state schema,
    langgraph's base ``ToolNode`` raises a ``KeyError`` if that field is absent
    from the state dict.

    This subclass overrides ``_inject_tool_args`` so that, when a specific state
    field is absent from the state dict:

    * If the tool parameter **has a default value**, the injection is skipped and
      the default is used instead — this is the common ``NotRequired`` case.
    * If the tool parameter has **no default**, the original ``KeyError`` is
      re-raised with a clearer message so the user is aware of the missing field.

    This matches the behaviour described in
    https://github.com/langchain-ai/langchain/issues/35585.
    """

    def _inject_tool_args(
        self,
        tool_call: ToolCall,
        tool_runtime: ToolRuntime,
        tool: BaseTool | None = None,
    ) -> ToolCall:
        injected = self._injected_args.get(tool_call["name"])
        if not injected and tool is not None:
            from langgraph.prebuilt.tool_node import _get_all_injected_args

            injected = _get_all_injected_args(tool)
        if not injected:
            return tool_call

        # If there are no state injections we can delegate entirely to the parent.
        if not injected.state:
            return super()._inject_tool_args(tool_call, tool_runtime, tool)

        state = tool_runtime.state

        # For dict state, check whether each requested field actually exists before
        # delegating to the parent.  Fields that are missing but have a default on
        # the tool function are simply skipped; fields that are missing and have no
        # default trigger a clear KeyError.
        if isinstance(state, dict):
            tool_obj = (
                tool
                if tool is not None
                else self._tools_by_name.get(tool_call["name"])
            )

            missing_no_default: list[str] = []
            missing_with_default: set[str] = set()

            for tool_arg, state_field in injected.state.items():
                if state_field is None:
                    # Injecting the full state object — always present.
                    continue
                if state_field not in state:
                    if tool_obj is not None and _tool_param_has_default(
                        tool_obj, tool_arg
                    ):
                        missing_with_default.add(tool_arg)
                    else:
                        missing_no_default.append(state_field)

            if missing_no_default:
                raise KeyError(
                    f"State field(s) {missing_no_default!r} are required by tool "
                    f"'{tool_call['name']}' via InjectedState but are absent from the "
                    "current state. Declare the field as NotRequired in your state "
                    "schema and give the tool parameter a default value, or ensure "
                    "the field is always populated before the tool is called."
                )

            if missing_with_default:
                # Build a patched tool_runtime whose state only contains the fields
                # that are actually present, then delegate to the parent.  The parent
                # will iterate over injected.state and read each field — for the ones
                # in missing_with_default the field won't be in the state dict, but
                # we need to make the parent skip them rather than KeyError.
                #
                # The simplest safe approach: inject all present fields ourselves and
                # return, mirroring what the parent does but skipping absent ones.
                tool_call_copy: ToolCall = copy(tool_call)
                injected_args: dict[str, Any] = {}

                for tool_arg, state_field in injected.state.items():
                    if tool_arg in missing_with_default:
                        # Field absent but has a default — skip injection.
                        continue
                    injected_args[tool_arg] = state[state_field] if state_field else state

                # Inject store if needed
                if injected.store:
                    if tool_runtime.store is None:
                        raise ValueError(
                            "Cannot inject store into tools with InjectedStore "
                            "annotations - please compile your graph with a store."
                        )
                    injected_args[injected.store] = tool_runtime.store

                # Inject runtime if needed
                if injected.runtime:
                    injected_args[injected.runtime] = tool_runtime

                tool_call_copy["args"] = {**tool_call_copy["args"], **injected_args}
                return tool_call_copy

        # No missing fields (or non-dict state) — delegate to parent for all other cases.
        return super()._inject_tool_args(tool_call, tool_runtime, tool)
