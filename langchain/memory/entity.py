from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
)
from langchain.memory.utils import get_prompt_input_key
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel, BaseMessage, get_buffer_string


class ConversationEntityMemory(BaseChatMemory, BaseModel):
    """Entity extractor & summarizer to memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT
    store: Dict[str, Optional[str]] = {}
    entity_cache: List[str] = []
    k: int = 3
    chat_history_key: str = "history"

    @property
    def buffer(self) -> List[BaseMessage]:
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return ["entities", self.chat_history_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        output = chain.predict(
            history=buffer_string,
            input=inputs[prompt_input_key],
        )
        if output.strip() == "NONE":
            entities = []
        else:
            entities = [w.strip() for w in output.split(",")]
        entity_summaries = {}
        for entity in entities:
            entity_summaries[entity] = self.store.get(entity, "")
        self.entity_cache = entities
        if self.return_messages:
            buffer: Any = self.buffer[-self.k * 2 :]
        else:
            buffer = buffer_string
        return {
            self.chat_history_key: buffer,
            "entities": entity_summaries,
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        input_data = inputs[prompt_input_key]
        chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)

        for entity in self.entity_cache:
            existing_summary = self.store.get(entity, "")
            output = chain.predict(
                summary=existing_summary,
                entity=entity,
                history=buffer_string,
                input=input_data,
            )
            self.store[entity] = output.strip()

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
        self.store = {}
