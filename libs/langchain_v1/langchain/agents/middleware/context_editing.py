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
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

DEFAULT_TOOL_PLACEHOLDER = "[cleared]"


TokenCounter = Callable[
    [Sequence[BaseMessage]],
    int,
]

ContextFraction = tuple[Literal["fraction"], float]
ContextTokens = tuple[Literal["tokens"], int]
ContextMessages = tuple[Literal["messages"], int]

ContextSize = ContextFraction | ContextTokens | ContextMessages


def _coerce_to_context_size(
    value: int | ContextSize, *, kind: Literal["trigger", "keep"], param_name: str
) -> ContextSize:
    """Coerce integer values to ContextSize tuples for backwards compatibility.

    Args:
        value: Integer or ContextSize tuple.
        kind: Whether this is for a trigger or keep parameter.
        param_name: Name of the parameter for deprecation warnings.

    Returns:
        ContextSize tuple.
    """
    if isinstance(value, int):
        # trigger uses tokens, keep uses messages (backwards compat with old API)
        context_type = "tokens" if kind == "trigger" else "messages"
        warnings.warn(
            f"{param_name}={value} (int) is deprecated. "
            f"Use {param_name}=('{context_type}', {value}) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return (context_type, value)
    return value


class ContextEdit(Protocol):
    """Protocol describing a context editing strategy."""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
        model_profile: Mapping[str, Any] | None = None,
    ) -> None:
        """Apply an edit to the message list in place."""
        ...


@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    """Configuration for clearing tool outputs when token limits are exceeded."""

    trigger: int | ContextSize | list[int | ContextSize] = ("tokens", 100_000)
    """One or more thresholds that trigger the edit.

    Provide a single `ContextSize` tuple or a list of tuples, in which case
    the edit runs when any threshold is breached.

    For backwards compatibility, integers are interpreted as token counts.

    Examples: `("messages", 50)`, `("tokens", 100_000)`, `100_000`,
        `[("fraction", 0.8), ("messages", 100)]`.
    """

    clear_at_least: int = 0
    """Minimum number of tokens to reclaim when the edit runs."""

    keep: int | ContextSize = ("messages", 3)
    """Context retention policy for tool results.

    Provide a `ContextSize` tuple to specify how many tool results to preserve.

    For backwards compatibility, integers are interpreted as message counts.

    Defaults to keeping the most recent 3 tool results.

    Examples: `("messages", 3)`, `3`, `("tokens", 3000)`, or `("fraction", 0.3)`.
    """

    clear_tool_inputs: bool = False
    """Whether to clear the originating tool call parameters on the AI message."""

    exclude_tools: Sequence[str] = ()
    """List of tool names to exclude from clearing."""

    placeholder: str = DEFAULT_TOOL_PLACEHOLDER
    """Placeholder text inserted for cleared tool outputs."""

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        # Coerce and validate trigger
        if isinstance(self.trigger, list):
            coerced_list = []
            for idx, item in enumerate(self.trigger):
                if isinstance(item, int):
                    coerced = _coerce_to_context_size(
                        item, kind="trigger", param_name=f"trigger[{idx}]"
                    )
                else:
                    coerced = item
                validated = self._validate_context_size(coerced, "trigger")
                coerced_list.append(validated)
            object.__setattr__(self, "trigger", coerced_list)
        else:
            if isinstance(self.trigger, int):
                coerced = _coerce_to_context_size(
                    self.trigger, kind="trigger", param_name="trigger"
                )
            else:
                coerced = self.trigger
            validated = self._validate_context_size(coerced, "trigger")
            object.__setattr__(self, "trigger", validated)

        # Coerce and validate keep
        if isinstance(self.keep, int):
            coerced_keep = _coerce_to_context_size(self.keep, kind="keep", param_name="keep")
        else:
            coerced_keep = self.keep
        validated_keep = self._validate_context_size(coerced_keep, "keep")
        object.__setattr__(self, "keep", validated_keep)

    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            # For keep, 0 is valid (means keep nothing)
            # For trigger, must be > 0
            min_value = 0 if parameter_name == "keep" else 1
            if value < min_value:
                msg = f"{parameter_name} thresholds must be >= {min_value}, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type {kind} for {parameter_name}."
            raise ValueError(msg)
        return context

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
        model_profile: Mapping[str, Any] | None = None,
    ) -> None:
        """Apply the clear-tool-uses strategy."""
        tokens = count_tokens(messages)

        if not self._should_trigger(messages, tokens, model_profile):
            return

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        # Calculate how many tool results to keep
        keep_count = self._calculate_keep_count(candidates, model_profile)

        if keep_count >= len(candidates):
            candidates = []
        elif keep_count > 0:
            candidates = candidates[:-keep_count]

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

    def _should_trigger(
        self,
        messages: list[AnyMessage],
        total_tokens: int,
        model_profile: Mapping[str, Any] | None,
    ) -> bool:
        """Determine whether the edit should run for the current context usage."""
        trigger_conditions = self.trigger if isinstance(self.trigger, list) else [self.trigger]

        for kind, value in trigger_conditions:
            if kind == "messages" and len(messages) >= value:
                return True
            if kind == "tokens" and total_tokens >= value:
                return True
            if kind == "fraction":
                max_input_tokens = self._get_profile_limits(model_profile)
                if max_input_tokens is None:
                    continue
                threshold = int(max_input_tokens * value)
                if threshold <= 0:
                    threshold = 1
                if total_tokens >= threshold:
                    return True
        return False

    def _calculate_keep_count(
        self,
        candidates: list[tuple[int, ToolMessage]],
        model_profile: Mapping[str, Any] | None,
    ) -> int:
        """Calculate how many tool results to keep based on retention policy."""
        kind, value = self.keep
        if kind == "messages":
            return int(value)
        if kind == "tokens":
            # For token-based retention, we would need to count tokens per tool message
            # For simplicity, convert to message count based on average
            # This is a simplified implementation - could be enhanced
            return int(value)
        if kind == "fraction":
            max_input_tokens = self._get_profile_limits(model_profile)
            if max_input_tokens is None:
                # Fallback to default message count
                return 3
            target_count = int(len(candidates) * value)
            if target_count <= 0:
                target_count = 1
            return target_count
        return 3  # Default fallback

    def _get_profile_limits(self, model_profile: Mapping[str, Any] | None) -> int | None:
        """Retrieve max input token limit from the model profile."""
        if model_profile is None:
            return None

        if not isinstance(model_profile, Mapping):
            return None

        max_input_tokens = model_profile.get("max_input_tokens")

        if not isinstance(max_input_tokens, int):
            return None

        return max_input_tokens

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

        # Validate that fractional limits can be used
        requires_profile = False
        for edit in self.edits:
            if isinstance(edit, ClearToolUsesEdit):
                trigger_conditions = (
                    edit.trigger if isinstance(edit.trigger, list) else [edit.trigger]
                )
                for condition in trigger_conditions:
                    if condition[0] == "fraction":
                        requires_profile = True
                        break
                if edit.keep[0] == "fraction":
                    requires_profile = True

        if requires_profile:
            # Just warn, don't raise - we'll handle it gracefully at runtime
            pass

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

        # Get model profile if available
        model_profile = self._get_model_profile(request.model)

        for edit in self.edits:
            edit.apply(request.messages, count_tokens=count_tokens, model_profile=model_profile)

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

        # Get model profile if available
        model_profile = self._get_model_profile(request.model)

        for edit in self.edits:
            edit.apply(request.messages, count_tokens=count_tokens, model_profile=model_profile)

        return await handler(request)

    def _get_model_profile(self, model: Any) -> Mapping[str, Any] | None:
        """Retrieve model profile if available."""
        try:
            profile = model.profile
        except (AttributeError, ImportError):
            return None

        if not isinstance(profile, Mapping):
            return None

        return profile


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
    "ContextFraction",
    "ContextMessages",
    "ContextSize",
    "ContextTokens",
]
