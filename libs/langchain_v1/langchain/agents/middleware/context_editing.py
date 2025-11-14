"""Context editing middleware.

Mirrors Anthropic's context editing capabilities by clearing older tool results once the
conversation grows beyond a configurable token threshold.

The implementation is intentionally model-agnostic so it can be used with any LangChain
chat model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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

if TYPE_CHECKING:
    from langchain.agents.middleware._context import ContextCondition, ContextSize

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
    """Configuration for clearing tool outputs when token limits are exceeded.

    Supports flexible trigger and keep configurations using `ContextSize` tuples or
    backwards-compatible integer values.
    """

    trigger: ContextCondition | int = 100_000
    """Trigger condition(s) for when the edit should run.

    Supports flexible AND/OR logic via nested lists:
    - Single condition: `("messages", 50)` or `("tokens", 3000)`
    - OR conditions: `[("tokens", 3000), ("messages", 100)]`
    - AND conditions: `[("tokens", 500), ("fraction", 0.8)]` as nested list
    - Mixed AND/OR: `[("messages", 10), [("tokens", 500), ("fraction", 0.8)]]`

    For backwards compatibility, also accepts an integer token count.
    """

    clear_at_least: int = 0
    """Minimum number of tokens to reclaim when the edit runs."""

    keep: ContextSize | int = 3
    """Context retention policy for tool results.

    Provide a `ContextSize` tuple to specify how much history to preserve:
    - `("messages", 3)` - Keep last 3 tool results
    - `("tokens", 1000)` - Keep tool results within token budget
    - `("fraction", 0.3)` - Keep tool results within 30% of model's max tokens

    For backwards compatibility, also accepts an integer message count.
    """

    clear_tool_inputs: bool = False
    """Whether to clear the originating tool call parameters on the AI message."""

    exclude_tools: Sequence[str] = ()
    """List of tool names to exclude from clearing."""

    placeholder: str = DEFAULT_TOOL_PLACEHOLDER
    """Placeholder text inserted for cleared tool outputs."""

    model: Any = None
    """Optional model instance for fractional token limits."""

    _trigger_conditions: list[ContextSize | list[ContextSize]] | None = None
    _keep_normalized: ContextSize | None = None
    _trigger_as_int: int | None = None
    _keep_as_int: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize trigger/keep parameters."""
        # Normalize trigger
        if isinstance(self.trigger, int):
            self._trigger_as_int = self.trigger
            self._trigger_conditions = None
        elif isinstance(self.trigger, tuple):
            # Single ContextSize
            self._validate_context_size(self.trigger, "trigger")
            self._trigger_conditions = [self.trigger]
            self._trigger_as_int = None
        elif isinstance(self.trigger, list):
            # List of conditions
            self._trigger_conditions = self._validate_trigger_conditions(self.trigger)
            self._trigger_as_int = None
        else:
            msg = f"trigger must be int or ContextCondition, got {type(self.trigger).__name__}"
            raise TypeError(msg)

        # Normalize keep
        if isinstance(self.keep, int):
            self._keep_as_int = self.keep
            self._keep_normalized = None
        elif isinstance(self.keep, tuple):
            self._validate_context_size(self.keep, "keep")
            self._keep_normalized = self.keep
            self._keep_as_int = None
        else:
            msg = f"keep must be int or ContextSize, got {type(self.keep).__name__}"
            raise TypeError(msg)

        # Check if model profile is required
        requires_profile = False
        if self._trigger_conditions:
            requires_profile = self._requires_profile(self._trigger_conditions)
        if self._keep_normalized and self._keep_normalized[0] == "fraction":
            requires_profile = True

        if requires_profile and self.model is None:
            msg = (
                "model parameter is required when using fractional token limits. "
                "Pass a model instance or use absolute token/message counts instead."
            )
            raise ValueError(msg)

        if requires_profile and self._get_profile_limits() is None:
            msg = (
                "Model profile information is required to use fractional token limits. "
                'pip install "langchain[model-profiles]" or use absolute token counts instead.'
            )
            raise ValueError(msg)

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply the clear-tool-uses strategy."""
        tokens = count_tokens(messages)

        if not self._should_trigger(messages, tokens):
            return

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        # Determine how many tool results to keep
        keep_count = self._determine_keep_count(messages, tokens)

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

    def _should_trigger(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether the edit should trigger based on current state."""
        # Backwards compatibility: int trigger
        if self._trigger_as_int is not None:
            return total_tokens > self._trigger_as_int

        # New API: ContextCondition with AND/OR logic
        if not self._trigger_conditions:
            return False

        # OR logic across top-level conditions
        for condition in self._trigger_conditions:
            if isinstance(condition, list):
                # AND group - all must be satisfied
                if self._check_and_group(condition, messages, total_tokens):
                    return True
            elif self._check_single_condition(condition, messages, total_tokens):
                # Single condition
                return True
        return False

    def _check_and_group(
        self, and_group: list[ContextSize], messages: list[AnyMessage], total_tokens: int
    ) -> bool:
        """Check if all conditions in an AND group are satisfied."""
        for condition in and_group:
            if not self._check_single_condition(condition, messages, total_tokens):
                return False
        return True

    def _check_single_condition(
        self, condition: ContextSize, messages: list[AnyMessage], total_tokens: int
    ) -> bool:
        """Check if a single condition is satisfied."""
        kind, value = condition
        if kind == "messages":
            return len(messages) >= value
        if kind == "tokens":
            return total_tokens >= value
        if kind == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return False
            threshold = int(max_input_tokens * value)
            if threshold <= 0:
                threshold = 1
            return total_tokens >= threshold
        return False

    def _determine_keep_count(self, messages: list[AnyMessage], total_tokens: int) -> int:  # noqa: ARG002
        """Determine how many tool results to keep based on keep configuration."""
        # Backwards compatibility: int keep
        if self._keep_as_int is not None:
            return self._keep_as_int

        # New API: ContextSize
        if self._keep_normalized is None:
            return 0

        kind, value = self._keep_normalized
        if kind == "messages":
            return int(value)
        if kind in {"tokens", "fraction"}:
            # For token-based keep, we need to count backwards through tool messages
            # to find how many fit within the budget
            target_tokens = self._get_target_token_count(value, kind)
            if target_tokens is None:
                return 0
            return self._count_tool_messages_within_budget(messages, target_tokens)
        return 0

    def _get_target_token_count(self, value: float, kind: str) -> int | None:
        """Get the target token count for token/fraction-based keep."""
        if kind == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return None
            target = int(max_input_tokens * value)
        elif kind == "tokens":
            target = int(value)
        else:
            return None

        return max(1, target) if target > 0 else 1

    def _count_tool_messages_within_budget(
        self, messages: list[AnyMessage], target_tokens: int
    ) -> int:
        """Count how many recent tool messages fit within token budget."""
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        if not tool_messages:
            return 0

        # Count backwards from the end
        count = 0
        accumulated_tokens = 0
        for tool_msg in reversed(tool_messages):
            # Approximate token count for this message
            msg_tokens = len(str(tool_msg.content))
            if accumulated_tokens + msg_tokens > target_tokens and count > 0:
                break
            accumulated_tokens += msg_tokens
            count += 1

        return count

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile."""
        if self.model is None:
            return None

        try:
            profile = self.model.profile
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
            # For trigger, value must be > 0. For keep, value can be >= 0 (0 means keep nothing)
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

    def _validate_trigger_conditions(
        self, conditions: list[Any]
    ) -> list[ContextSize | list[ContextSize]]:
        """Validate and normalize trigger conditions with nested AND/OR logic.

        Args:
            conditions: List of ContextSize tuples or nested lists of ContextSize tuples.

        Returns:
            Validated list where top-level items are OR'd and nested lists are AND'd.
        """
        validated: list[ContextSize | list[ContextSize]] = []
        for item in conditions:
            if isinstance(item, tuple):
                # Single condition (tuple)
                validated.append(self._validate_context_size(item, "trigger"))
            elif isinstance(item, list):
                # AND group (nested list)
                if not item:
                    msg = "Empty AND groups are not allowed in trigger conditions."
                    raise ValueError(msg)
                and_group = [self._validate_context_size(cond, "trigger") for cond in item]
                validated.append(and_group)
            else:
                msg = f"Trigger conditions must be tuples or lists, got {type(item).__name__}."
                raise ValueError(msg)
        return validated

    def _requires_profile(self, conditions: list[ContextSize | list[ContextSize]]) -> bool:
        """Check if any condition requires model profile information."""
        for condition in conditions:
            if isinstance(condition, list):
                # AND group
                if any(c[0] == "fraction" for c in condition):
                    return True
            elif condition[0] == "fraction":
                return True
        return False


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
