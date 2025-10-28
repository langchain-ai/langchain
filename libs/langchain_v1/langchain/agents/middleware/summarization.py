"""Summarization middleware."""

import uuid
from collections.abc import Callable, Iterable
from typing import Any, cast
import warnings

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
from langgraph.runtime import Runtime

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.chat_models import BaseChatModel, init_chat_model

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step. Because of this, you must do your very best to extract and record all of the most important context from the conversation history.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.
</instructions>

The user will message you with the full message history you'll be extracting context from, to then replace. Carefully read over it all, and think deeply about what information is most important to your overall goal that should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501

SUMMARY_PREFIX = "## Previous conversation summary:"

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15
_SEARCH_RANGE_FOR_TOOL_PAIRS = 5


class SummarizationMiddleware(AgentMiddleware):
    """Middleware that summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        max_tokens_before_summary: int | None = None,
        messages_to_keep: int = _DEFAULT_MESSAGES_TO_KEEP,
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        summary_prefix: str = SUMMARY_PREFIX,
        *,
        target_retention_pct: float = 0.3,
        buffer_tokens: int = 500,
    ) -> None:
        """Initialize the summarization middleware.

        Args:
            model: The language model to use for generating summaries.
            max_tokens_before_summary: Token threshold to trigger summarization.
                If `None` and model capabilities are unavailable, summarization is disabled.
                Deprecated in favor of dynamic token management using model capabilities.
            messages_to_keep: Number of recent messages to preserve after summarization.
                Used as fallback when token-based calculation is not possible.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            summary_prefix: Prefix added to system message when including summary.
            target_retention_pct: Target percentage of max_input_tokens to retain after
                summarization (default: 0.3 or 30%). Only used when model capabilities
                are available via langchain-llm-features.
            buffer_tokens: Safety buffer in tokens to prevent hitting limits
                (default: 500). Only used when model capabilities are available via
                langchain-llm-features.
        """
        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
        self.max_tokens_before_summary = max_tokens_before_summary
        self.messages_to_keep = messages_to_keep
        self.token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.summary_prefix = summary_prefix
        self.target_retention_pct = target_retention_pct
        self.buffer_tokens = buffer_tokens

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Process messages before model invocation, potentially triggering summarization."""
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)

        # Try to use model capabilities for dynamic token management
        should_summarize = False
        target_token_count: int | None = None

        try:
            max_input_tokens = self.model.capabilities.get("max_input_tokens")
            max_output_tokens = self.model.capabilities.get("max_output_tokens")

            if max_input_tokens is not None and max_output_tokens is not None:
                if total_tokens + max_output_tokens + self.buffer_tokens > max_input_tokens:
                    should_summarize = True
                    target_token_count = int(max_input_tokens * self.target_retention_pct)
                else:
                    return None
        except ImportError:
                warning_msg = (
                    "Defaulting to None max_tokens_before_summary. pip install "
                    "langchain-llm-capabilities to enable setting based on model "
                    "context window size."
                )
                warnings.warn(warning_msg, ImportWarning)

        if (
            self.max_tokens_before_summary is not None
            and total_tokens < self.max_tokens_before_summary
            and not should_summarize
        ):
            return None

        cutoff_index = self._find_safe_cutoff(messages, target_token_count=target_token_count)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary = self._create_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    def _build_new_messages(self, summary: str) -> list[HumanMessage]:
        return [
            HumanMessage(content=f"Here is a summary of the conversation to date:\n\n{summary}")
        ]

    def _ensure_message_ids(self, messages: list[AnyMessage]) -> None:
        """Ensure all messages have unique IDs for the add_messages reducer."""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

    def _partition_messages(
        self,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        messages_to_summarize = conversation_messages[:cutoff_index]
        preserved_messages = conversation_messages[cutoff_index:]

        return messages_to_summarize, preserved_messages

    def _find_safe_cutoff(
        self,
        messages: list[AnyMessage],
        target_token_count: int | None = None,
    ) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Args:
            messages: List of messages to find a cutoff point for.
            target_token_count: Target token count to retain. If provided, keeps as many
                messages as possible while staying under this limit. If `None`, falls back
                to keeping `messages_to_keep` recent messages.

        Returns:
            The index where messages can be safely cut without separating
            related AI and Tool messages. Returns 0 if no safe cutoff is found.
        """
        if target_token_count is not None:
            # Token-based cutoff: use trim_messages to keep as many as possible
            try:
                trimmed = trim_messages(
                    messages,
                    max_tokens=target_token_count,
                    token_counter=self.token_counter,
                    strategy="last",
                    allow_partial=False,
                )
                # Find where the trimmed messages start in the original list
                if not trimmed:
                    return 0

                # Calculate cutoff index based on how many messages were kept
                num_kept = len(trimmed)
                target_cutoff = len(messages) - num_kept

                # Ensure we have a safe cutoff point (doesn't separate AI/Tool pairs)
                for i in range(target_cutoff, -1, -1):
                    if self._is_safe_cutoff_point(messages, i):
                        return i

                return 0
            except Exception:  # noqa: BLE001
                # Fall back to message-count-based approach if trim_messages fails
                pass

        # Legacy message-count-based cutoff
        if len(messages) <= self.messages_to_keep:
            return 0

        target_cutoff = len(messages) - self.messages_to_keep

        for i in range(target_cutoff, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0

    def _is_safe_cutoff_point(self, messages: list[AnyMessage], cutoff_index: int) -> bool:
        """Check if cutting at index would separate AI/Tool message pairs."""
        if cutoff_index >= len(messages):
            return True

        search_start = max(0, cutoff_index - _SEARCH_RANGE_FOR_TOOL_PAIRS)
        search_end = min(len(messages), cutoff_index + _SEARCH_RANGE_FOR_TOOL_PAIRS)

        for i in range(search_start, search_end):
            if not self._has_tool_calls(messages[i]):
                continue

            tool_call_ids = self._extract_tool_call_ids(cast("AIMessage", messages[i]))
            if self._cutoff_separates_tool_pair(messages, i, cutoff_index, tool_call_ids):
                return False

        return True

    def _has_tool_calls(self, message: AnyMessage) -> bool:
        """Check if message is an AI message with tool calls."""
        return (
            isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls  # type: ignore[return-value]
        )

    def _extract_tool_call_ids(self, ai_message: AIMessage) -> set[str]:
        """Extract tool call IDs from an AI message."""
        tool_call_ids = set()
        for tc in ai_message.tool_calls:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if call_id is not None:
                tool_call_ids.add(call_id)
        return tool_call_ids

    def _cutoff_separates_tool_pair(
        self,
        messages: list[AnyMessage],
        ai_message_index: int,
        cutoff_index: int,
        tool_call_ids: set[str],
    ) -> bool:
        """Check if cutoff separates an AI message from its corresponding tool messages."""
        for j in range(ai_message_index + 1, len(messages)):
            message = messages[j]
            if isinstance(message, ToolMessage) and message.tool_call_id in tool_call_ids:
                ai_before_cutoff = ai_message_index < cutoff_index
                tool_before_cutoff = j < cutoff_index
                if ai_before_cutoff != tool_before_cutoff:
                    return True
        return False

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages."""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        try:
            response = self.model.invoke(self.summary_prompt.format(messages=trimmed_messages))
            return cast("str", response.content).strip()
        except Exception as e:  # noqa: BLE001
            return f"Error generating summary: {e!s}"

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits."""
        try:
            return trim_messages(
                messages,
                max_tokens=_DEFAULT_TRIM_TOKEN_LIMIT,
                token_counter=self.token_counter,
                start_on="human",
                strategy="last",
                allow_partial=True,
                include_system=True,
            )
        except Exception:  # noqa: BLE001
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
