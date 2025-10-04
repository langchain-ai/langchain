"""Context editing middleware.

This middleware mirrors Anthropic's context editing capabilities by clearing
older tool results once the conversation grows beyond a configurable token
threshold. The implementation is intentionally model-agnostic so it can be used
with any LangChain chat model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import Protocol

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

DEFAULT_TOOL_PLACEHOLDER = "[cleared]"


TokenCounter = Callable[
    [Sequence[BaseMessage]],
    int,
]


class ContextEdit(Protocol):
    """Protocol describing a context editing strategy."""

    def apply(
        self,
        *,
        tokens: int,
        messages: list[AnyMessage],
        count_tokens: TokenCounter,
    ) -> int:
        """Apply an edit to the message list, returning the new token count."""
        ...


@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    """Configuration for clearing tool outputs when token limits are exceeded."""

    trigger_tokens: int = 100000
    """Token count that triggers the edit."""

    clear_at_least: int = 0
    """Minimum number of tokens to reclaim when the edit runs."""

    keep: int = 3
    """Number of most recent tool results that must be preserved."""

    clear_tool_inputs: bool = False
    """Whether to clear the originating tool call parameters on the AI message."""

    exclude_tools: Sequence[str] = ()
    """List of tool names to exclude from clearing."""

    placeholder: str = DEFAULT_TOOL_PLACEHOLDER
    """Placeholder text inserted for cleared tool outputs."""

    def apply(
        self,
        *,
        tokens: int,
        messages: list[AnyMessage],
        count_tokens: TokenCounter,
    ) -> int:
        """Apply the clear-tool-uses strategy."""
        if tokens <= self.trigger_tokens:
            return tokens

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if self.keep >= len(candidates):
            candidates = []
        elif self.keep:
            candidates = candidates[: -self.keep]

        cleared_tokens = 0
        excluded_tools = set(self.exclude_tools)

        for idx, tool_message in candidates:
            if self.clear_at_least > 0 and cleared_tokens >= self.clear_at_least:
                break

            if tool_message.response_metadata.get("context_editing", {}).get("cleared"):
                continue

            ai_message = next(
                (m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)), None
            )

            if ai_message is None:
                continue

            tool_call = next(
                (
                    call
                    for call in ai_message.tool_calls
                    if call.get("id") == tool_message.tool_call_id
                ),
                None,
            )

            if tool_call is None:
                continue

            if (tool_message.name or tool_call["name"]) in excluded_tools:
                continue

            messages[idx] = tool_message.model_copy(
                update={
                    "artifact": None,
                    "content": self.placeholder,
                    "response_metadata": {
                        **tool_message.response_metadata,
                        "context_editing": {
                            "cleared": True,
                            "strategy": "clear_tool_uses",
                        },
                    },
                }
            )

            if self.clear_tool_inputs:
                messages[messages.index(ai_message)] = self._build_cleared_tool_input_message(
                    ai_message,
                    tool_message.tool_call_id,
                )

            new_token_count = count_tokens(messages)
            cleared_tokens = max(0, tokens - new_token_count)

        return tokens - cleared_tokens

    def _build_cleared_tool_input_message(
        self,
        message: AIMessage,
        tool_call_id: str,
    ) -> AIMessage:
        updated_tool_calls = []
        cleared_any = False
        for tool_call in message.tool_calls:
            updated_call = dict(tool_call)
            if updated_call.get("id") == tool_call_id:
                updated_call["args"] = {}
                cleared_any = True
            updated_tool_calls.append(updated_call)

        metadata = dict(getattr(message, "response_metadata", {}))
        context_entry = dict(metadata.get("context_editing", {}))
        if cleared_any:
            cleared_ids = set(context_entry.get("cleared_tool_inputs", []))
            cleared_ids.add(tool_call_id)
            context_entry["cleared_tool_inputs"] = sorted(cleared_ids)
            metadata["context_editing"] = context_entry

        return message.model_copy(
            update={
                "tool_calls": updated_tool_calls,
                "response_metadata": metadata,
            }
        )


class ContextEditingMiddleware(AgentMiddleware):
    """Middleware that automatically prunes tool results to manage context size.

    The middleware applies a sequence of edits when the total input token count
    exceeds configured thresholds. Currently the ``ClearToolUsesEdit`` strategy is
    supported, aligning with Anthropic's ``clear_tool_uses_20250919`` behaviour.
    """

    edits: list[ContextEdit]
    token_count_method: Literal["approx", "model"]

    def __init__(
        self,
        *,
        edits: Iterable[ContextEdit] | None = None,
        token_count_method: Literal["approx", "model"] = "approx",  # noqa: S107
    ) -> None:
        """Initialise a context editing middleware instance.

        Args:
            edits: Sequence of edit strategies to apply. Defaults to a single
                `ClearToolUsesEdit` mirroring Anthropic defaults.
            token_count_method: Whether to use approximate token counting
                (faster, less accurate) or exact counting implemented by the
                chat model (potentially slower, more accurate).
        """
        super().__init__()
        self.edits = list(edits or (ClearToolUsesEdit(),))
        self.token_count_method = token_count_method

    def modify_model_request(
        self,
        request: ModelRequest,
        state: AgentState,  # noqa: ARG002
        runtime: Runtime,  # noqa: ARG002
    ) -> ModelRequest:
        """Modify the model request by applying context edits before invocation."""
        if not request.messages:
            return request

        if self.token_count_method == "approx":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)
        else:
            system_msg = (
                [SystemMessage(content=request.system_prompt)] if request.system_prompt else []
            )

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(system_msg + list(messages))

        tokens = count_tokens(request.messages)

        for edit in self.edits:
            tokens = edit.apply(
                tokens=tokens,
                messages=request.messages,
                count_tokens=count_tokens,
            )

        return request


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
]
