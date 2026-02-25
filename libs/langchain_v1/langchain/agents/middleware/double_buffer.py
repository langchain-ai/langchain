"""Double-buffered context window middleware.

Implements proactive context compaction using double buffering — instead of
stop-the-world summarization when context fills, this middleware begins
summarizing at a configurable threshold while the agent continues working,
then swaps to the pre-built back buffer seamlessly.

"""

import asyncio
import logging
import uuid
from enum import Enum
from functools import partial
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    get_buffer_string,
    trim_messages,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.middleware.summarization import (
    ContextSize,
    TokenCounter,
    _get_approximate_token_counter,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langchain.chat_models import BaseChatModel, init_chat_model

logger = logging.getLogger(__name__)

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15

DOUBLE_BUFFER_SOURCE = "double_buffer"

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Extract the most important context from the conversation history
below for a seamless context transition.
</primary_objective>

<instructions>
You are creating a checkpoint summary for a double-buffered context
window. The conversation will continue seamlessly after this summary
replaces older messages. Ensure no critical information is lost.

Structure your summary using these sections:

## SESSION INTENT
What is the user's primary goal? What overall task is being accomplished?

## SUMMARY
Extract all important context: key decisions, conclusions,
strategies, rejected options and reasoning.

## ARTIFACTS
Files, resources, or artifacts created/modified/accessed. Include specific file paths and changes.

## NEXT STEPS
What tasks remain? What should happen next?

</instructions>

<messages>
Messages to summarize:
{messages}
</messages>"""


class RenewalPolicy(str, Enum):
    """Policy for handling accumulated compression debt across generations."""

    RECURSE = "recurse"
    """Summarize the accumulated summaries (meta-compression)."""

    DUMP = "dump"
    """Discard all summaries and start fresh."""


class DoubleBufferMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Double-buffered context window management middleware.

    Instead of stop-the-world compaction when context fills, this middleware:

    1. **Checkpoint** at a configurable threshold — summarize and seed a back buffer
    2. **Concurrent** — keep working, the back buffer accumulates new messages
    3. **Swap** — when the active buffer hits the wall, swap to the pre-built back buffer

    Summaries accumulate across generations up to ``max_generations`` before
    triggering renewal (meta-summarization or clean restart).

    Example:
        ```python
        from langchain.agents.middleware import DoubleBufferMiddleware

        middleware = DoubleBufferMiddleware(
            model="anthropic:claude-sonnet-4-20250514",
            checkpoint_trigger=("fraction", 0.7),
            swap_trigger=("fraction", 0.95),
            keep=("messages", 20),
            max_generations=5,
        )
        ```
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        checkpoint_trigger: ContextSize = ("fraction", 0.7),
        swap_trigger: ContextSize = ("fraction", 0.95),
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        max_generations: int | None = None,
        renewal_policy: RenewalPolicy = RenewalPolicy.RECURSE,
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        checkpoint_timeout: float = 120.0,
    ) -> None:
        """Initialize double buffer middleware.

        Args:
            model: The language model to use for generating summaries.
            checkpoint_trigger: Threshold that triggers checkpoint summarization
                and back buffer creation.
            swap_trigger: Threshold that triggers swapping to the back buffer.
                Must represent a higher capacity than ``checkpoint_trigger``.
            keep: Context retention policy — how much recent history to preserve
                in the back buffer after summarization.
            max_generations: Maximum number of summary-on-summary layers before
                triggering renewal. None means no limit (renewal disabled).
            renewal_policy: How to handle accumulated compression debt when
                ``max_generations`` is reached.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Maximum tokens to keep when preparing
                messages for the summarization call. Pass ``None`` to skip trimming.
            checkpoint_timeout: Maximum seconds to wait for a background
                checkpoint task before cancelling it.
        """
        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
        self.checkpoint_trigger = self._validate_context_size(
            checkpoint_trigger, "checkpoint_trigger"
        )
        self.swap_trigger = self._validate_context_size(swap_trigger, "swap_trigger")
        self._validate_trigger_ordering(self.checkpoint_trigger, self.swap_trigger)
        self.keep = self._validate_context_size(keep, "keep")
        self.max_generations = max_generations
        self.renewal_policy = renewal_policy
        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize
        self.checkpoint_timeout = checkpoint_timeout

        if token_counter is count_tokens_approximately:
            self.token_counter = _get_approximate_token_counter(self.model)
            self._partial_token_counter: TokenCounter = partial(  # type: ignore[call-arg]
                self.token_counter, use_usage_metadata_scaling=False
            )
        else:
            self.token_counter = token_counter
            self._partial_token_counter = token_counter

        # Internal state
        self._back_buffer: list[AnyMessage] | None = None
        self._current_generation: int = 0
        self._checkpoint_active: bool = False
        self._checkpoint_task: asyncio.Task[list[AnyMessage] | None] | None = None

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Process messages before model invocation.

        Implements the three-phase double buffer algorithm:
        - Phase 1: Create checkpoint and back buffer at checkpoint_trigger
        - Phase 2: (implicit) Back buffer accumulates new messages
        - Phase 3: Swap to back buffer at swap_trigger

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            An updated state with swapped messages if a swap occurred.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)
        total_tokens = self.token_counter(messages)

        # Phase 3: Swap if we've hit the swap threshold
        if self._should_trigger(messages, total_tokens, self.swap_trigger):
            if self._back_buffer is not None:
                return self._perform_swap(messages)
            # No back buffer at swap time — fall back to stop-the-world checkpoint
            logger.info(
                "Swap threshold reached with no back buffer. "
                "Falling back to synchronous checkpoint."
            )
            back_buffer = self._create_checkpoint(messages)
            if back_buffer is not None:
                return self._perform_swap(messages)
            logger.warning("Checkpoint failed at swap time. Continuing with full context.")

        # Phase 1: Create checkpoint if we've hit the checkpoint threshold
        if (
            self._back_buffer is None
            and not self._checkpoint_active
            and self._should_trigger(messages, total_tokens, self.checkpoint_trigger)
        ):
            self._create_checkpoint(messages)
            return None

        # Phase 2: Concurrent — update back buffer with any new messages
        if self._back_buffer is not None:
            self._sync_back_buffer(messages)

        return None

    @override
    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async version of before_model.

        Uses ``asyncio.create_task`` for non-blocking checkpoint creation.
        The agent continues working while the summary is generated in the
        background. If the swap threshold is reached before the checkpoint
        finishes, it blocks on the task (graceful degradation to
        stop-the-world). If no checkpoint was ever started, it falls back
        to a synchronous checkpoint at swap time.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            An updated state with swapped messages if a swap occurred.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)
        total_tokens = self.token_counter(messages)

        # Reap completed background checkpoint task
        if self._checkpoint_task is not None and self._checkpoint_task.done():
            try:
                self._checkpoint_task.result()  # surface any exceptions
            except Exception:
                logger.warning("Background checkpoint task failed.", exc_info=True)
            self._checkpoint_task = None

        # Phase 3: Swap
        if self._should_trigger(messages, total_tokens, self.swap_trigger):
            # If background checkpoint is still running, block on it with timeout
            if self._checkpoint_task is not None and not self._checkpoint_task.done():
                logger.info(
                    "Swap threshold hit while async checkpoint still running. "
                    "Blocking on checkpoint (timeout=%.1fs).",
                    self.checkpoint_timeout,
                )
                try:
                    await asyncio.wait_for(self._checkpoint_task, timeout=self.checkpoint_timeout)
                except TimeoutError:
                    logger.warning(
                        "Background checkpoint timed out after %.1fs. Cancelling.",
                        self.checkpoint_timeout,
                    )
                    self._checkpoint_task.cancel()
                except Exception:
                    logger.warning(
                        "Background async checkpoint failed at swap time.",
                        exc_info=True,
                    )
                finally:
                    self._checkpoint_task = None

            if self._back_buffer is not None:
                return self._perform_swap(messages)

            # No back buffer — fall back to stop-the-world checkpoint
            logger.info(
                "Swap threshold reached with no back buffer. "
                "Falling back to synchronous checkpoint."
            )
            back_buffer = await self._acreate_checkpoint(messages)
            if back_buffer is not None:
                return self._perform_swap(messages)
            logger.warning("Checkpoint failed at swap time. Continuing with full context.")

        # Phase 1: Checkpoint — kick off background summarization
        if (
            self._back_buffer is None
            and not self._checkpoint_active
            and self._should_trigger(messages, total_tokens, self.checkpoint_trigger)
        ):
            self._checkpoint_task = asyncio.create_task(self._acreate_checkpoint(messages))
            return None

        # Phase 2: Concurrent — sync back buffer with new messages
        if self._back_buffer is not None:
            self._sync_back_buffer(messages)

        return None

    def _should_trigger(
        self, messages: list[AnyMessage], total_tokens: int, trigger: ContextSize
    ) -> bool:
        """Check if a trigger condition is met."""
        kind, value = trigger
        if kind == "messages":
            return len(messages) >= value
        if kind == "tokens":
            return total_tokens >= value
        # The remaining case is "fraction".
        max_input_tokens = self._get_profile_limits()
        if max_input_tokens is None:
            return False
        threshold = int(max_input_tokens * value)
        return total_tokens >= max(threshold, 1)

    def _create_checkpoint(self, messages: list[AnyMessage]) -> list[AnyMessage] | None:
        """Phase 1: Summarize and create the back buffer (sync).

        Returns:
            The seeded back buffer on success, or ``None`` on failure.
        """
        self._checkpoint_active = True
        back_buffer: list[AnyMessage] | None = None

        try:
            # Handle generation renewal if needed
            max_gens = self.max_generations
            if max_gens is not None and self._current_generation >= max_gens:
                self._perform_renewal(messages)

            cutoff_index = self._determine_cutoff_index(messages)
            if cutoff_index <= 0:
                self._checkpoint_active = False
                return None

            messages_to_summarize, preserved_messages = self._partition_messages(
                messages, cutoff_index
            )
            summary = self._create_summary(messages_to_summarize)
            summary_messages = self._build_summary_messages(summary)

            # Seed back buffer: summary + preserved recent messages
            back_buffer = [*summary_messages, *preserved_messages]
            self._back_buffer = back_buffer
            self._checkpoint_active = False

            logger.info(
                "Double-buffer checkpoint created (generation %d). Back buffer: %d messages.",
                self._current_generation + 1,
                len(back_buffer),
            )
        except Exception:
            self._checkpoint_active = False
            logger.warning(
                "Double-buffer checkpoint creation failed.",
                exc_info=True,
            )
        return back_buffer

    async def _acreate_checkpoint(self, messages: list[AnyMessage]) -> list[AnyMessage] | None:
        """Phase 1: Summarize and create the back buffer (async).

        Returns:
            The seeded back buffer on success, or ``None`` on failure.
        """
        self._checkpoint_active = True
        back_buffer: list[AnyMessage] | None = None

        try:
            max_gens = self.max_generations
            if max_gens is not None and self._current_generation >= max_gens:
                await self._aperform_renewal(messages)

            cutoff_index = self._determine_cutoff_index(messages)
            if cutoff_index <= 0:
                self._checkpoint_active = False
                return None

            messages_to_summarize, preserved_messages = self._partition_messages(
                messages, cutoff_index
            )
            summary = await self._acreate_summary(messages_to_summarize)
            summary_messages = self._build_summary_messages(summary)

            back_buffer = [*summary_messages, *preserved_messages]
            self._back_buffer = back_buffer
            self._checkpoint_active = False

            logger.info(
                "Double-buffer checkpoint created (generation %d). Back buffer: %d messages.",
                self._current_generation + 1,
                len(back_buffer),
            )
        except Exception:
            self._checkpoint_active = False
            logger.warning(
                "Double-buffer checkpoint creation failed.",
                exc_info=True,
            )
        return back_buffer

    def _perform_swap(self, messages: list[AnyMessage]) -> dict[str, Any]:
        """Phase 3: Swap the back buffer into the active context."""
        back_buffer = self._back_buffer or []
        self._back_buffer = None
        self._current_generation += 1
        self._checkpoint_active = False

        logger.info(
            "Double-buffer swap (generation %d). Active: %d -> %d messages.",
            self._current_generation,
            len(messages),
            len(back_buffer),
        )

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *back_buffer,
            ]
        }

    def _sync_back_buffer(self, messages: list[AnyMessage]) -> None:
        """Phase 2: Keep back buffer in sync with new messages.

        Appends any messages from the active buffer that aren't yet in
        the back buffer, identified by message ID.
        """
        if self._back_buffer is None:
            return

        back_ids = {msg.id for msg in self._back_buffer if msg.id}
        for msg in messages:
            if msg.id and msg.id not in back_ids:
                self._back_buffer.append(msg)
                back_ids.add(msg.id)

    def _perform_renewal(self, messages: list[AnyMessage]) -> None:
        """Handle accumulated compression debt (sync)."""
        logger.info(
            "Max generations (%d) reached. Renewal policy: %s",
            self.max_generations,
            self.renewal_policy.value,
        )

        if self.renewal_policy == RenewalPolicy.DUMP:
            self._current_generation = 0
            return

        # RECURSE: meta-summarize existing summaries
        summary_msgs: list[AnyMessage] = [
            msg
            for msg in messages
            if isinstance(msg, HumanMessage)
            and msg.additional_kwargs.get("lc_source") == DOUBLE_BUFFER_SOURCE
        ]
        if summary_msgs:
            self._create_summary(summary_msgs)
            logger.info("Meta-summarization completed for renewal.")
        self._current_generation = 0

    async def _aperform_renewal(self, messages: list[AnyMessage]) -> None:
        """Handle accumulated compression debt (async)."""
        logger.info(
            "Max generations (%d) reached. Renewal policy: %s",
            self.max_generations,
            self.renewal_policy.value,
        )

        if self.renewal_policy == RenewalPolicy.DUMP:
            self._current_generation = 0
            return

        summary_msgs: list[AnyMessage] = [
            msg
            for msg in messages
            if isinstance(msg, HumanMessage)
            and msg.additional_kwargs.get("lc_source") == DOUBLE_BUFFER_SOURCE
        ]
        if summary_msgs:
            await self._acreate_summary(summary_msgs)
            logger.info("Meta-summarization completed for renewal.")
        self._current_generation = 0

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        kind, value = self.keep
        if kind in {"tokens", "fraction"}:
            cutoff = self._find_token_based_cutoff(messages)
            if cutoff is not None:
                return cutoff
            return self._find_safe_cutoff(messages, _DEFAULT_MESSAGES_TO_KEEP)
        return self._find_safe_cutoff(messages, cast("int", value))

    def _find_token_based_cutoff(self, messages: list[AnyMessage]) -> int | None:
        """Find cutoff index based on target token retention."""
        if not messages:
            return 0

        kind, value = self.keep
        if kind == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return None
            target_token_count = int(max_input_tokens * value)
        elif kind == "tokens":
            target_token_count = int(value)
        else:
            return None

        if target_token_count <= 0:
            target_token_count = 1

        if self.token_counter(messages) <= target_token_count:
            return 0

        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1
        for _ in range(max_iterations):
            if left >= right:
                break
            mid = (left + right) // 2
            if self._partial_token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left
        if cutoff_candidate >= len(messages):
            cutoff_candidate = max(len(messages) - 1, 0)

        return self._find_safe_cutoff_point(messages, cutoff_candidate)

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile."""
        try:
            profile = self.model.profile
        except AttributeError:
            return None

        if not isinstance(profile, dict):
            return None

        max_input_tokens = profile.get("max_input_tokens")
        return max_input_tokens if isinstance(max_input_tokens, int) else None

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages (sync)."""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed:
            return "Previous conversation was too long to summarize."

        formatted = get_buffer_string(trimmed)
        response = self.model.invoke(
            self.summary_prompt.format(messages=formatted).rstrip(),
            config={"metadata": {"lc_source": DOUBLE_BUFFER_SOURCE}},
        )
        return response.text.strip()

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages (async)."""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed:
            return "Previous conversation was too long to summarize."

        formatted = get_buffer_string(trimmed)
        response = await self.model.ainvoke(
            self.summary_prompt.format(messages=formatted).rstrip(),
            config={"metadata": {"lc_source": DOUBLE_BUFFER_SOURCE}},
        )
        return response.text.strip()

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits."""
        try:
            if self.trim_tokens_to_summarize is None:
                return messages
            return cast(
                "list[AnyMessage]",
                trim_messages(
                    messages,
                    max_tokens=self.trim_tokens_to_summarize,
                    token_counter=self.token_counter,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]

    @staticmethod
    def _validate_context_size(context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            if value <= 0:
                msg = f"{parameter_name} thresholds must be greater than 0, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type {kind} for {parameter_name}."
            raise ValueError(msg)
        return context

    @staticmethod
    def _validate_trigger_ordering(
        checkpoint_trigger: ContextSize, swap_trigger: ContextSize
    ) -> None:
        """Validate that swap_trigger is strictly greater than checkpoint_trigger.

        When both triggers use the same unit type, the swap threshold must
        exceed the checkpoint threshold. Otherwise the back buffer is never
        ready before swap fires, silently degrading to repeated
        stop-the-world fallback.

        Args:
            checkpoint_trigger: The checkpoint trigger configuration.
            swap_trigger: The swap trigger configuration.

        Raises:
            ValueError: If both triggers share the same type and
                swap_trigger is not strictly greater than checkpoint_trigger.
        """
        cp_kind, cp_value = checkpoint_trigger
        sw_kind, sw_value = swap_trigger
        if cp_kind == sw_kind and sw_value <= cp_value:
            msg = (
                f"swap_trigger ({sw_kind}, {sw_value}) must be strictly greater "
                f"than checkpoint_trigger ({cp_kind}, {cp_value}). Otherwise the "
                f"back buffer will never be ready before swap fires, causing "
                f"repeated stop-the-world fallback."
            )
            raise ValueError(msg)

    @staticmethod
    def _build_summary_messages(summary: str) -> list[HumanMessage]:
        """Build summary messages for the back buffer."""
        return [
            HumanMessage(
                content=f"Here is a summary of the conversation to date:\n\n{summary}",
                additional_kwargs={"lc_source": DOUBLE_BUFFER_SOURCE},
            )
        ]

    @staticmethod
    def _ensure_message_ids(messages: list[AnyMessage]) -> None:
        """Ensure all messages have unique IDs."""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

    @staticmethod
    def _partition_messages(
        messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        return messages[:cutoff_index], messages[cutoff_index:]

    @staticmethod
    def _find_safe_cutoff(messages: list[AnyMessage], messages_to_keep: int) -> int:
        """Find safe cutoff that preserves AI/Tool message pairs."""
        if len(messages) <= messages_to_keep:
            return 0
        target = len(messages) - messages_to_keep
        return DoubleBufferMiddleware._find_safe_cutoff_point(messages, target)

    @staticmethod
    def _find_safe_cutoff_point(messages: list[AnyMessage], cutoff_index: int) -> int:
        """Find safe cutoff point that doesn't split AI/Tool message pairs."""
        if cutoff_index >= len(messages) or not isinstance(messages[cutoff_index], ToolMessage):
            return cutoff_index

        tool_call_ids: set[str] = set()
        idx = cutoff_index
        while idx < len(messages) and isinstance(messages[idx], ToolMessage):
            tool_msg = cast("ToolMessage", messages[idx])
            if tool_msg.tool_call_id:
                tool_call_ids.add(tool_msg.tool_call_id)
            idx += 1

        for i in range(cutoff_index - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                ai_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                if tool_call_ids & ai_tool_call_ids:
                    return i

        return idx

    @property
    def generation(self) -> int:
        """Current generation count (number of buffer swaps completed)."""
        return self._current_generation

    @property
    def has_back_buffer(self) -> bool:
        """Whether a back buffer is currently active."""
        return self._back_buffer is not None
