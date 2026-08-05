"""Context editing middleware.

Mirrors Anthropic's context editing capabilities by clearing older tool results once the
conversation grows beyond a configurable token threshold.

The implementation is intentionally model-agnostic so it can be used with any LangChain
chat model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command
from typing_extensions import Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

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


class ContextEditingMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Automatically prune tool results to manage context size.

    The middleware applies a sequence of edits when the total input token count exceeds
    configured thresholds. Messages an edit changes are persisted back to agent
    state (via a `Command` update), so a configured checkpointer observes the
    same cleared tool results the model saw, and subsequent turns' idempotency
    checks (e.g. `ClearToolUsesEdit`'s `cleared` flag) see accurate history.

    Currently the `ClearToolUsesEdit` strategy is supported, aligning with Anthropic's
    `clear_tool_uses_20250919` behavior [(read more)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).
    """

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
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The result of invoking the handler with potentially edited messages.
            When an edit changes any message, the changed messages are also
            returned as a `Command` update so the checkpointer persists them
            (see `_with_persisted_edits`).
        """
        if not request.messages:
            return handler(request)

        count_tokens = self._make_token_counter(request)

        edited_messages = list(request.messages)
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        response = handler(request.override(messages=edited_messages))
        return self._with_persisted_edits(response, request.messages, edited_messages)

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
            The result of invoking the handler with potentially edited messages.
            When an edit changes any message, the changed messages are also
            returned as a `Command` update so the checkpointer persists them
            (see `_with_persisted_edits`).
        """
        if not request.messages:
            return await handler(request)

        count_tokens = self._make_token_counter(request)

        edited_messages = list(request.messages)
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        response = await handler(request.override(messages=edited_messages))
        return self._with_persisted_edits(response, request.messages, edited_messages)

    def _make_token_counter(self, request: ModelRequest[ContextT]) -> TokenCounter:
        """Build the token counter for `request`, per `self.token_count_method`."""
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

    @staticmethod
    def _with_persisted_edits(
        response: ModelResponse[ResponseT],
        original_messages: Sequence[AnyMessage],
        edited_messages: Sequence[AnyMessage],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
        """Wrap `response` with a `Command` persisting any messages edits changed.

        `edited_messages` is built from a shallow copy of `original_messages`,
        so edits that don't touch a given message leave that slot referencing
        the identical object; only messages `ClearToolUsesEdit.apply` actually
        replaced (via `model_copy`) differ by identity. Persisting exactly
        those messages — decided using the same `count_tokens` as the request
        that was actually sent to the model — keeps the checkpointer's view
        consistent with what the model saw, regardless of `token_count_method`.
        Without this, a persistence pass that recomputed edits with a
        different counter could disagree with `wrap_model_call`'s decision
        and either skip persisting a clear that was applied, or persist one
        that wasn't.
        """
        changed = [
            new_msg
            for old_msg, new_msg in zip(original_messages, edited_messages, strict=True)
            if new_msg is not old_msg
        ]
        if not changed:
            return response
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"messages": changed}),
        )


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
]
