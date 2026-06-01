"""Context editing middleware.

Mirrors Anthropic's context editing capabilities by clearing older tool results once the
conversation grows beyond a configurable token threshold.

The implementation is intentionally model-agnostic so it can be used with any LangChain
chat model.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command
from typing_extensions import NotRequired, Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)

logger = logging.getLogger(__name__)

DEFAULT_TOOL_PLACEHOLDER = "[cleared]"


TokenCounter = Callable[
    [Sequence[BaseMessage]],
    int,
]


class ContextEdit(Protocol):
    """Protocol describing a context editing strategy."""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply an edit to the message list in place."""
        ...


@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    """Configuration for clearing tool outputs when token limits are exceeded."""

    trigger: int = 100_000
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
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply the clear-tool-uses strategy."""
        tokens = count_tokens(messages)

        if tokens <= self.trigger:
            return

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

            if self.clear_at_least > 0:
                new_token_count = count_tokens(messages)
                cleared_tokens = max(0, tokens - new_token_count)
                if cleared_tokens >= self.clear_at_least:
                    break

        return

    @staticmethod
    def _build_cleared_tool_input_message(
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


class ContextEditingState(AgentState[ResponseT]):
    """State schema for `ContextEditingMiddleware`.

    Extends `AgentState` with a persisted measure of the post-edit effective token
    count.

    Edits run on a throwaway copy of the messages and are never written back to the
    checkpoint, so under a persistent checkpointer every turn reloads the full,
    uncleared history. `_last_effective_count` records how many tokens the
    conversation occupied *after* the most recent edit, giving downstream logic a
    stable, checkpoint-backed signal it can read to drive escalation (e.g.
    summarization) when clearing alone cannot keep the conversation under budget.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

    _last_effective_count: NotRequired[Annotated[int, PrivateStateAttr]]


def _merge_update(command: Command[Any] | None, update: dict[str, Any]) -> Command[Any]:
    """Merge `update` into an existing command's update mapping.

    Args:
        command: An existing command returned by the handler, or `None`.
        update: The state update to merge in.

    Returns:
        A command whose `update` mapping includes `update`. If the existing command
        carries a non-mapping update (which the `wrap_model_call` command contract does
        not produce), it is returned untouched to avoid clobbering it.
    """
    if command is None:
        return Command(update=update)
    existing = command.update
    if existing is None:
        return replace(command, update=update)
    if isinstance(existing, dict):
        return replace(command, update={**existing, **update})
    return command


class ContextEditingMiddleware(
    AgentMiddleware[ContextEditingState[ResponseT], ContextT, ResponseT]
):
    """Automatically prune tool results to manage context size.

    The middleware applies a sequence of edits when the total input token count exceeds
    configured thresholds.

    Currently the `ClearToolUsesEdit` strategy is supported, aligning with Anthropic's
    `clear_tool_uses_20250919` behavior [(read more)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).

    Edits are applied to a throwaway copy of the messages on every model call, so the
    cleared placeholders are never written back to the checkpoint. To keep eviction
    decisions stable across turns when a persistent checkpointer reloads the full
    history, the middleware persists the post-edit effective token count to
    `ContextEditingState._last_effective_count` via a `Command` returned alongside the
    model response.
    """

    state_schema = ContextEditingState  # type: ignore[assignment]

    edits: list[ContextEdit]
    token_count_method: Literal["approximate", "model"]

    def __init__(
        self,
        *,
        edits: Iterable[ContextEdit] | None = None,
        token_count_method: Literal["approximate", "model"] = "approximate",  # noqa: S107
    ) -> None:
        """Initialize an instance of context editing middleware.

        Args:
            edits: Sequence of edit strategies to apply.

                Defaults to a single `ClearToolUsesEdit` mirroring Anthropic defaults.
            token_count_method: Whether to use approximate token counting
                (faster, less accurate) or exact counting implemented by the
                chat model (potentially slower, more accurate).
        """
        super().__init__()
        self.edits = list(edits or (ClearToolUsesEdit(),))
        self.token_count_method = token_count_method

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Apply context edits before invoking the model via handler.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The handler result wrapped in an `ExtendedModelResponse` whose command
            persists the post-edit effective token count.
        """
        if not request.messages:
            return handler(request)

        edited_messages, effective_count = self._apply_edits(request)
        result = handler(request.override(messages=edited_messages))
        return self._persist_effective_count(result, effective_count)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Apply context edits before invoking the model via handler.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The handler result wrapped in an `ExtendedModelResponse` whose command
            persists the post-edit effective token count.
        """
        if not request.messages:
            return await handler(request)

        edited_messages, effective_count = self._apply_edits(request)
        result = await handler(request.override(messages=edited_messages))
        return self._persist_effective_count(result, effective_count)

    def _resolve_token_counter(self, request: ModelRequest[ContextT]) -> TokenCounter:
        """Build the token counter for the configured counting method."""
        if self.token_count_method == "approximate":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)

        else:
            system_msg = [request.system_message] if request.system_message else []

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(
                    system_msg + list(messages), request.tools
                )

        return count_tokens

    def _apply_edits(self, request: ModelRequest[ContextT]) -> tuple[list[AnyMessage], int]:
        """Apply edits to a copy of the messages and measure the post-edit token count.

        Edits run on a deep copy so the original (checkpointed) messages are never
        mutated. The returned count is the effective token footprint the model will
        actually see.

        Args:
            request: Model request whose messages should be edited.

        Returns:
            A tuple of the edited messages and their effective token count.
        """
        count_tokens = self._resolve_token_counter(request)

        edited_messages = deepcopy(list(request.messages))
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        effective_count = count_tokens(edited_messages)
        self._warn_if_over_budget(effective_count)
        return edited_messages, effective_count

    def _warn_if_over_budget(self, effective_count: int) -> None:
        """Warn when edits could not bring the conversation under the tightest trigger.

        Args:
            effective_count: Post-edit effective token count.
        """
        triggers = [
            trigger
            for edit in self.edits
            if (trigger := getattr(edit, "trigger", None)) is not None
        ]
        if not triggers:
            return

        tightest = min(triggers)
        if effective_count > tightest:
            logger.warning(
                "Context editing left %d tokens, which still exceeds the configured "
                "trigger of %d. Clearing tool results was insufficient; consider "
                "escalating (e.g. summarization) to keep the conversation under budget.",
                effective_count,
                tightest,
            )

    def _persist_effective_count(
        self,
        result: ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT],
        effective_count: int,
    ) -> ExtendedModelResponse[ResponseT]:
        """Attach a command persisting the effective token count to the handler result.

        Args:
            result: The value returned by the handler.
            effective_count: Post-edit effective token count to persist.

        Returns:
            An `ExtendedModelResponse` whose command updates `_last_effective_count`.
        """
        update: dict[str, Any] = {"_last_effective_count": effective_count}
        if isinstance(result, ExtendedModelResponse):
            model_response = result.model_response
            command = _merge_update(result.command, update)
        elif isinstance(result, AIMessage):
            model_response = ModelResponse(result=[result])
            command = Command(update=update)
        else:
            model_response = result
            command = Command(update=update)

        return ExtendedModelResponse(model_response=model_response, command=command)


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
]
