"""Utils file included for backwards compat imports."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from langgraph.prebuilt import InjectedState, InjectedStore, ToolRuntime
from langgraph.prebuilt.tool_node import (
    ToolCallRequest,
    ToolCallWithContext,
    ToolCallWrapper,
    _get_all_injected_args,
)
from langgraph.prebuilt.tool_node import (
    ToolNode as _ToolNode,
)

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall
    from langchain_core.tools import BaseTool


class ToolNode(_ToolNode):
    """ToolNode subclass that gracefully handles ``NotRequired`` state fields.

    Upstream ``langgraph.prebuilt.ToolNode._inject_tool_args`` accesses state
    fields with ``state[field]`` which raises ``KeyError`` when the field is
    declared as ``NotRequired`` in the state schema and is absent at runtime.

    This subclass overrides ``_inject_tool_args`` to use ``.get()`` (for dict
    state) and ``getattr(…, None)`` (for object state) so that missing optional
    fields resolve to ``None`` instead of crashing.

    See: https://github.com/langchain-ai/langchain/issues/35585
    """

    def _inject_tool_args(
        self,
        tool_call: ToolCall,
        tool_runtime: ToolRuntime,
        tool: BaseTool | None = None,
    ) -> ToolCall:
        injected = self._injected_args.get(tool_call["name"])
        if not injected and tool is not None:
            injected = _get_all_injected_args(tool)
        if not injected:
            return tool_call

        tool_call_copy: ToolCall = copy(tool_call)
        injected_args: dict = {}

        # Inject state
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
                        f"Invalid input to ToolNode. "
                        f"Tool {tool_call['name']} requires "
                        f"graph state dict as input."
                    )
                    if any(
                        state_field for state_field in injected.state.values()
                    ):
                        required_fields_str = ", ".join(
                            f for f in required_fields if f
                        )
                        err_msg += (
                            f" State should contain fields "
                            f"{required_fields_str}."
                        )
                    raise ValueError(err_msg)

            # Extract state values — use .get() / getattr default so that
            # NotRequired fields that are absent resolve to None (#35585).
            if isinstance(state, dict):
                for tool_arg, state_field in injected.state.items():
                    injected_args[tool_arg] = (
                        state.get(state_field) if state_field else state
                    )
            else:
                for tool_arg, state_field in injected.state.items():
                    injected_args[tool_arg] = (
                        getattr(state, state_field, None)
                        if state_field
                        else state
                    )

        # Inject store
        if injected.store:
            if tool_runtime.store is None:
                msg = (
                    "Cannot inject store into tools with InjectedStore "
                    "annotations - please compile your graph with a store."
                )
                raise ValueError(msg)
            injected_args[injected.store] = tool_runtime.store

        # Inject runtime
        if injected.runtime:
            injected_args[injected.runtime] = tool_runtime

        tool_call_copy["args"] = {**tool_call_copy["args"], **injected_args}
        return tool_call_copy


__all__ = [
    "InjectedState",
    "InjectedStore",
    "ToolCallRequest",
    "ToolCallWithContext",
    "ToolCallWrapper",
    "ToolNode",
    "ToolRuntime",
]
