"""Utils file included for backwards compat imports."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

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

    Keep the override as narrow as possible: delegate to upstream
    ``langgraph.prebuilt.ToolNode._inject_tool_args`` by default, and only
    recover the specific case where an injected state field is optional and
    absent at runtime. That keeps LangChain aligned with upstream ToolNode
    changes while still fixing ``#35585``.

    See: https://github.com/langchain-ai/langchain/issues/35585
    """

    def _inject_tool_args(
        self,
        tool_call: ToolCall,
        tool_runtime: ToolRuntime,
        tool: BaseTool | None = None,
    ) -> ToolCall:
        try:
            return super()._inject_tool_args(tool_call, tool_runtime, tool=tool)
        except (KeyError, AttributeError) as err:
            injected = self._injected_args.get(tool_call["name"])
            if not injected and tool is not None:
                injected = _get_all_injected_args(tool)
            if not injected or not injected.state:
                raise

            state: Any = tool_runtime.state
            if isinstance(state, dict):
                missing_optional_state = any(
                    state_field and state_field not in state
                    for state_field in injected.state.values()
                )
                if not missing_optional_state:
                    raise
                injected_args: dict[str, Any] = {
                    tool_arg: state.get(state_field) if state_field else state
                    for tool_arg, state_field in injected.state.items()
                }
            else:
                missing_optional_state = any(
                    state_field and not hasattr(state, state_field)
                    for state_field in injected.state.values()
                )
                if not missing_optional_state:
                    raise
                injected_args = {
                    tool_arg: getattr(state, state_field, None) if state_field else state
                    for tool_arg, state_field in injected.state.items()
                }

            tool_call_copy: ToolCall = copy(tool_call)

            if injected.store:
                if tool_runtime.store is None:
                    msg = (
                        "Cannot inject store into tools with InjectedStore "
                        "annotations - please compile your graph with a store."
                    )
                    raise ValueError(msg) from err
                injected_args[injected.store] = tool_runtime.store

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
