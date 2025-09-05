import uuid
from collections.abc import Callable, Iterable, Sequence

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]


from langchain.agents.types import AgentMiddleware, AgentState

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
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context."""


class SummarizationMiddleware(AgentMiddleware):
    def __init__(
        self,
        model: LanguageModelLike,
        max_tokens_before_summary: int | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        messages_to_leave: int = 20,
        summary_system_prompt: str = DEFAULT_SUMMARY_PROMPT,
        fake_tool_call_name: str = "summarize_convo",
    ):
        super().__init__()
        self.model = model
        self.max_tokens_before_summary = max_tokens_before_summary
        self.token_counter = token_counter
        self.messages_to_leave = messages_to_leave
        self.summary_system_prompt = summary_system_prompt
        self.fake_tool_call_name = fake_tool_call_name

    def before_model(self, state: AgentState) -> AgentState | None:
        messages = state["messages"]
        token_counts = self.token_counter(messages)
        # If token counts are less than max allowed, then end this hook early
        if token_counts < self.max_tokens_before_summary:
            return None
        # Otherwise, we create a summary!
        # Get messages that we want to create a summary for
        messages_to_summarize = messages[: -self.messages_to_leave]
        # Create summary text
        summary = self._summarize_messages(messages_to_summarize)
        # Create fake messages to add to history
        fake_tool_call_id = str(uuid.uuid4())
        fake_messages = [
            AIMessage(
                content="Looks like I'm running out of tokens. I'm going to summarize the conversation history to free up space.",
                tool_calls={
                    "id": fake_tool_call_id,
                    "name": self.fake_tool_call_name,
                    "args": {
                        "reasoning": "I'm running out of tokens. I'm going to summarize all of the messages since my last summary message to free up space.",
                    },
                },
            ),
            ToolMessage(tool_call_id=fake_tool_call_id, content=summary),
        ]
        return {"messages": [RemoveMessage(id=m.id) for m in messages_to_summarize] + fake_messages}

    def _summarize_messages(self, messages_to_summarize: Sequence[AnyMessage]) -> str:
        system_message = self.summary_system_prompt
        user_message = self._format_messages(messages_to_summarize)
        response = self.model.invoke(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        # Use new .text attribute when ready
        return response.content

    @staticmethod
    def _format_messages(messages_to_summarize: Sequence[AnyMessage]) -> str:
        # TODO: better formatting logic
        return "\n".join([m.content for m in messages_to_summarize])
