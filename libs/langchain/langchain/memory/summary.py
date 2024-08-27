from __future__ import annotations

from typing import Any, Dict, List, Type

from langchain_core._api import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import pre_init

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import SUMMARY_PROMPT


@deprecated(
    since="0.2.12",
    removal="1.0",
    message=(
        "Refer here for how to incorporate summaries of conversation history: "
        "https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/"  # noqa: E501
    ),
)
class SummarizerMixin(BaseModel):
    """Mixin for summarizer."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    summary_message_cls: Type[BaseMessage] = SystemMessage

    def predict_new_summary(
        self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)

    async def apredict_new_summary(
        self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return await chain.apredict(summary=existing_summary, new_lines=new_lines)


class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    """Conversation summarizer to chat memory."""

    buffer: str = ""
    memory_key: str = "history"  #: :meta private:

    @classmethod
    def from_messages(
        cls,
        llm: BaseLanguageModel,
        chat_memory: BaseChatMessageHistory,
        *,
        summarize_step: int = 2,
        **kwargs: Any,
    ) -> ConversationSummaryMemory:
        obj = cls(llm=llm, chat_memory=chat_memory, **kwargs)
        for i in range(0, len(obj.chat_memory.messages), summarize_step):
            obj.buffer = obj.predict_new_summary(
                obj.chat_memory.messages[i : i + summarize_step], obj.buffer
            )
        return obj

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        if self.return_messages:
            buffer: Any = [self.summary_message_cls(content=self.buffer)]
        else:
            buffer = self.buffer
        return {self.memory_key: buffer}

    @pre_init
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-2:], self.buffer
        )

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.buffer = ""
