import uuid
from collections.abc import Callable, Iterable
from typing import cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

from langchain.agents.types import AgentMiddleware, AgentState
from langchain.chat_models import BaseChatModel

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
</messages>"""

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
        model: BaseChatModel,
        max_tokens_before_summary: int | None = None,
        messages_to_keep: int = _DEFAULT_MESSAGES_TO_KEEP,
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        summary_prefix: str = SUMMARY_PREFIX,
    ):
        """Initialize the summarization middleware.

        Args:
            model: The language model to use for generating summaries.
            max_tokens_before_summary: Token threshold to trigger summarization.
                If None, summarization is disabled.
            messages_to_keep: Number of recent messages to preserve after summarization.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            summary_prefix: Prefix added to system message when including summary.
        """
        super().__init__()
        self.model = model
        self.max_tokens_before_summary = max_tokens_before_summary
        self.messages_to_keep = messages_to_keep
        self.token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.summary_prefix = summary_prefix

    def before_model(self, state: AgentState) -> AgentState | None:
        """Process messages before model invocation, potentially triggering summarization."""
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if (
            self.max_tokens_before_summary is not None
            and total_tokens < self.max_tokens_before_summary
        ):
            return None

        system_message, conversation_messages = self._split_system_message(messages)
        cutoff_index = self._find_safe_cutoff(conversation_messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(
            system_message, conversation_messages, cutoff_index
        )

        summary = self._create_summary(messages_to_summarize)
        updated_system_message = self._build_updated_system_message(system_message, summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                updated_system_message,
                *preserved_messages,
            ]
        }

    def _ensure_message_ids(self, messages: list[AnyMessage]) -> None:
        """Ensure all messages have unique IDs for the add_messages reducer."""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

    def _split_system_message(
        self, messages: list[AnyMessage]
    ) -> tuple[SystemMessage | None, list[AnyMessage]]:
        """Separate system message from conversation messages."""
        if messages and isinstance(messages[0], SystemMessage):
            return messages[0], messages[1:]
        return None, messages

    def _partition_messages(
        self,
        system_message: SystemMessage | None,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve.

        We include the system message so that we can capture previous summaries.
        """
        messages_to_summarize = conversation_messages[:cutoff_index]
        preserved_messages = conversation_messages[cutoff_index:]

        if system_message is not None:
            messages_to_summarize = [system_message, *messages_to_summarize]

        return messages_to_summarize, preserved_messages

    def _build_updated_system_message(
        self, original_system_message: SystemMessage | None, summary: str
    ) -> SystemMessage:
        """Build new system message incorporating the summary."""
        if original_system_message is None:
            original_content = ""
        else:
            content = cast("str", original_system_message.content)
            original_content = content.split(self.summary_prefix)[0].strip()

        if original_content:
            content = f"{original_content}\n{self.summary_prefix}\n{summary}"
        else:
            content = f"{self.summary_prefix}\n{summary}"

        return SystemMessage(
            content=content,
            id=original_system_message.id if original_system_message else str(uuid.uuid4()),
        )

    def _find_safe_cutoff(self, messages: list[AnyMessage]) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Returns 0 if no safe cutoff is found.
        """
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

            tool_call_ids = self._extract_tool_call_ids(messages[i])
            if self._cutoff_separates_tool_pair(messages, i, cutoff_index, tool_call_ids):
                return False

        return True

    def _has_tool_calls(self, message: AnyMessage) -> bool:
        """Check if message is an AI message with tool calls."""
        return (
            isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls
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
        except Exception as e:
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
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
