"""Summarization middleware."""

import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final, cast

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

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15
_SEARCH_RANGE_FOR_TOOL_PAIRS = 5


class _UnsetType:
    """Sentinel indicating that max tokens should be inferred from the model profile."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _UnsetType()


class SummarizationMiddleware(AgentMiddleware):
    """Summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        tokens_before_summary: float | None = None,
        messages_before_summary: int | None = None,
        messages_to_keep: int | None = None,
        tokens_to_keep: float | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        **kwargs: Any,
    ) -> None:
        """Initialize the summarization middleware.

        Args:
            model: The language model to use for generating summaries.
            tokens_before_summary: Token threshold to trigger summarization.
                If `None`, token-based summarization is disabled.
                If >= 1, treats as an absolute token count threshold.
                If a float between 0 and 1 and model profile information is available,
                triggers summarization at `max_input_tokens * tokens_before_summary`.
            messages_before_summary: Message count threshold to trigger summarization.
                If `None`, message-based summarization is disabled.
            messages_to_keep: Number of recent messages to preserve after summarization.
                Cannot be specified together with `tokens_to_keep`. If neither tokens_to_keep
                nor messages_to_keep is specified, defaults to preserving 20 messages.
            tokens_to_keep: Number of tokens to preserve after summarization.
                If a float between 0 and 1, retains `max_input_tokens * tokens_to_keep`.
                If >= 1, treats as an absolute token count to retain.
                Cannot be specified together with `messages_to_keep`.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Maximum tokens to keep when preparing messages for the
                summarization call. Pass `None` to skip trimming entirely (risking
                summary model overflows if the history is too long).
        """
        super().__init__()

        # Handle backwards compatibility for max_tokens_before_summary
        if "max_tokens_before_summary" in kwargs:
            if tokens_before_summary is not None:
                msg = "Cannot specify both 'tokens_before_summary' and 'max_tokens_before_summary'"
                raise ValueError(msg)
            tokens_before_summary = kwargs.pop("max_tokens_before_summary")

        if messages_to_keep is not None and tokens_to_keep is not None:
            msg = "Cannot specify both 'messages_to_keep' and 'tokens_to_keep'"
            raise ValueError(msg)

        if messages_to_keep is None and tokens_to_keep is None:
            messages_to_keep = _DEFAULT_MESSAGES_TO_KEEP

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
        self.tokens_before_summary = tokens_before_summary
        self.messages_before_summary = messages_before_summary
        self.messages_to_keep = messages_to_keep
        self.tokens_to_keep = tokens_to_keep
        self.token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize

        if (
            (tokens_to_keep is not None and 0 < tokens_to_keep < 1)
            or (tokens_before_summary is not None and 0 < tokens_before_summary < 1)
        ) and self._get_profile_limits() is None:
            msg = (
                "Model profile information is required to use fractional token limits. "
                'pip install "langchain[model-profiles]" or use absolute token counts '
                "instead."
            )
            raise ValueError(msg)

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Process messages before model invocation, potentially triggering summarization."""
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)

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

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether summarization should run for the current token usage."""
        if self.messages_before_summary is not None and (
            len(messages) >= self.messages_before_summary
        ):
            return True

        if self.tokens_before_summary is None:
            return False

        # Handle float (fraction of max_input_tokens)
        if 0 < self.tokens_before_summary < 1:
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return False
            threshold = int(max_input_tokens * self.tokens_before_summary)
            return total_tokens > threshold

        # Handle int (absolute token count)
        return total_tokens >= self.tokens_before_summary

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        if self.tokens_to_keep is not None:
            token_based_cutoff = self._find_token_based_cutoff(messages)
            if token_based_cutoff is not None:
                return token_based_cutoff
        return self._find_safe_cutoff(messages)

    def _find_token_based_cutoff(self, messages: list[AnyMessage]) -> int | None:
        """Find cutoff index based on target token retention."""
        if not messages:
            return 0

        tokens_to_keep = cast("float | int", self.tokens_to_keep)
        if 0 < tokens_to_keep < 1:
            # fraction of max_input_tokens, if available
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return None
            target_token_count = int(max_input_tokens * tokens_to_keep)
        else:
            # absolute token count
            target_token_count = int(tokens_to_keep)

        if target_token_count <= 0:
            target_token_count = 1

        if self.token_counter(messages) <= target_token_count:
            return 0

        # Use binary search to identify the earliest message index that keeps the
        # suffix within the token budget.
        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1
        for _ in range(max_iterations):
            if left >= right:
                break

            mid = (left + right) // 2
            if self.token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left

        if cutoff_candidate >= len(messages):
            if len(messages) == 1:
                return 0
            cutoff_candidate = len(messages) - 1

        for i in range(cutoff_candidate, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile."""
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

    def _find_safe_cutoff(self, messages: list[AnyMessage]) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Returns 0 if no safe cutoff is found.
        """
        messages_to_keep = self.messages_to_keep or _DEFAULT_MESSAGES_TO_KEEP
        if len(messages) <= messages_to_keep:
            return 0

        target_cutoff = len(messages) - messages_to_keep

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
            return response.text.strip()
        except Exception as e:  # noqa: BLE001
            return f"Error generating summary: {e!s}"

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits."""
        try:
            if self.trim_tokens_to_summarize is None:
                return messages
            return trim_messages(
                messages,
                max_tokens=self.trim_tokens_to_summarize,
                token_counter=self.token_counter,
                start_on="human",
                strategy="last",
                allow_partial=True,
                include_system=True,
            )
        except Exception:  # noqa: BLE001
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
