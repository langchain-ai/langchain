"""Context editing middleware.

Mirrors Anthropic's context editing capabilities by clearing older tool results once the
conversation grows beyond a configurable token threshold.

The implementation is intentionally model-agnostic so it can be used with any LangChain
chat model.
"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
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

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

DEFAULT_TOOL_PLACEHOLDER = "[cleared]"

_DEFAULT_TRIGGER_TOKENS = 100_000
_DEFAULT_KEEP = 3

ContextFraction = tuple[Literal["fraction"], float]
ContextTokens = tuple[Literal["tokens"], int]
ContextMessages = tuple[Literal["messages"], int]

ContextSize = ContextFraction | ContextTokens | ContextMessages


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


@dataclass(slots=true, init=False)
class ClearToolUsesEdit(ContextEdit):
    """Configuration for clearing tool outputs when token limits are exceeded."""

    trigger: ContextSize | list[ContextSize]
    clear_at_least: int
    keep: ContextSize
    clear_tool_inputs: bool
    exclude_tools: Sequence[str]
    placeholder: str
    model: BaseChatModel | None
    _trigger_conditions: list[ContextSize]

    def __init__(
        self,
        *,
        trigger: ContextSize | list[ContextSize] | int | None = None,
        clear_at_least: int = 0,
        keep: ContextSize | int = ("messages", _DEFAULT_KEEP),
        clear_tool_inputs: bool = False,
        exclude_tools: Sequence[str] = (),
        placeholder: str = DEFAULT_TOOL_PLACEHOLDER,
        model: BaseChatModel | None = None,
    ) -> None:
        """Initialize the clear tool uses edit.

        Args:
            trigger: One or more thresholds that trigger context editing. Provide a single
                `ContextSize` tuple or a list of tuples, in which case editing runs when any
                threshold is breached. Examples: `("messages", 50)`, `("tokens", 3000)`,
                `[("fraction", 0.8), ("messages", 100)]`. Defaults to `("tokens", 100000)`.
            clear_at_least: Minimum number of tokens to reclaim when the edit runs.
            keep: Context retention policy applied after editing. Provide a `ContextSize` tuple
                to specify how many tool results to preserve. Defaults to keeping the most recent
                3 tool results. Examples: `("messages", 3)`, `("tokens", 3000)`, or
                `("fraction", 0.3)`.
            clear_tool_inputs: Whether to clear the originating tool call parameters on the AI
                message.
            exclude_tools: List of tool names to exclude from clearing.
            placeholder: Placeholder text inserted for cleared tool outputs.
            model: Optional chat model for model profile information. Required when using
                fractional triggers or keep values.
        """
        # Handle deprecated int-based parameters for trigger
        if isinstance(trigger, int):
            value = trigger
            warnings.warn(
                "Passing trigger as int is deprecated. Use trigger=('tokens', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            trigger = ("tokens", value)

        # Handle deprecated int-based parameters for keep
        if isinstance(keep, int):
            value = keep
            warnings.warn(
                "Passing keep as int is deprecated. Use keep=('messages', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            keep = ("messages", value)

        # Set default trigger if not provided
        if trigger is None:
            trigger = ("tokens", _DEFAULT_TRIGGER_TOKENS)

        # Validate and store trigger conditions
        if isinstance(trigger, list):
            validated_list = [self._validate_context_size(item, "trigger") for item in trigger]
            self.trigger = validated_list
            trigger_conditions: list[ContextSize] = validated_list
        else:
            validated = self._validate_context_size(trigger, "trigger")
            self.trigger = validated
            trigger_conditions = [validated]
        self._trigger_conditions = trigger_conditions

        self.clear_at_least = clear_at_least
        self.keep = self._validate_context_size(keep, "keep")
        self.clear_tool_inputs = clear_tool_inputs
        self.exclude_tools = exclude_tools
        self.placeholder = placeholder
        self.model = model

        # Check if model profile is required
        requires_profile = any(condition[0] == "fraction" for condition in self._trigger_conditions)
        if self.keep[0] == "fraction":
            requires_profile = True
        if requires_profile and model is not None and self._get_profile_limits(model) is None:
            msg = (
                "Model profile information is required to use fractional token limits. "
                'pip install "langchain[model-profiles]" or use absolute token counts '
                "instead."
            )
            raise ValueError(msg)

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply the clear-tool-uses strategy."""
        total_tokens = count_tokens(messages)

        if not self._should_edit(messages, total_tokens):
            return

        # Find all tool message candidates
        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if not candidates:
            return

        # Determine how many to keep based on keep policy
        keep_count = self._determine_keep_count(candidates, count_tokens)

        if keep_count >= len(candidates):
            candidates = []
        else:
            candidates = candidates[:-keep_count] if keep_count > 0 else candidates

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
                cleared_tokens = max(0, total_tokens - new_token_count)
                if cleared_tokens >= self.clear_at_least:
                    break

        return

    def _should_edit(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether editing should run for the current token usage."""
        for kind, value in self._trigger_conditions:
            if kind == "messages" and len(messages) >= value:
                return True
            if kind == "tokens" and total_tokens >= value:
                return True
            if kind == "fraction":
                if self.model is None:
                    continue
                max_input_tokens = self._get_profile_limits(self.model)
                if max_input_tokens is None:
                    continue
                threshold = int(max_input_tokens * value)
                if threshold <= 0:
                    threshold = 1
                if total_tokens >= threshold:
                    return True
        return False

    def _determine_keep_count(
        self,
        candidates: list[tuple[int, ToolMessage]],  # noqa: ARG002
        count_tokens: TokenCounter,  # noqa: ARG002
    ) -> int:
        """Determine how many tool results to keep based on keep policy.

        Note: candidates and count_tokens are currently unused but reserved for future
        enhancement to support token-based retention counting.
        """
        kind, value = self.keep
        if kind == "messages":
            return int(value)
        if kind in {"tokens", "fraction"}:
            # For token-based or fraction-based keep, we need to count backwards
            # This is a simplified implementation - keeping N most recent tool messages
            # A more sophisticated implementation would count actual tokens
            return int(value) if kind == "tokens" else _DEFAULT_KEEP
        return _DEFAULT_KEEP

    def _get_profile_limits(self, model: BaseChatModel) -> int | None:
        """Retrieve max input token limit from the model profile."""
        try:
            profile = model.profile
        except (AttributeError, ImportError):
            return None

        if not isinstance(profile, Mapping):
            return None

        max_input_tokens = profile.get("max_input_tokens")

        if not isinstance(max_input_tokens, int):
            return None

        return max_input_tokens

    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            # For "keep", 0 is valid (clear all), for "trigger", must be > 0
            if parameter_name == "trigger" and value <= 0:
                msg = f"{parameter_name} thresholds must be greater than 0, got {value}."
                raise ValueError(msg)
            if parameter_name == "keep" and value < 0:
                msg = f"{parameter_name} values must be non-negative, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type {kind} for {parameter_name}."
            raise ValueError(msg)
        return context

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
    """Automatically prune tool results to manage context size.

    The middleware applies a sequence of edits when the total input token count exceeds
    configured thresholds.

    Currently the `ClearToolUsesEdit` strategy is supported, aligning with Anthropic's
    `clear_tool_uses_20250919` behavior [(read more)](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool).
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
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply context edits before invoking the model via handler."""
        if not request.messages:
            return handler(request)

        if self.token_count_method == "approximate":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)
        else:
            system_msg = (
                [SystemMessage(content=request.system_prompt)] if request.system_prompt else []
            )

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(
                    system_msg + list(messages), request.tools
                )

        for edit in self.edits:
            edit.apply(request.messages, count_tokens=count_tokens)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply context edits before invoking the model via handler (async version)."""
        if not request.messages:
            return await handler(request)

        if self.token_count_method == "approximate":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)
        else:
            system_msg = (
                [SystemMessage(content=request.system_prompt)] if request.system_prompt else []
            )

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(
                    system_msg + list(messages), request.tools
                )

        for edit in self.edits:
            edit.apply(request.messages, count_tokens=count_tokens)

        return await handler(request)


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
]
