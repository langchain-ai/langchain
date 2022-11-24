"""Memory modules for conversation prompts."""
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from langchain.chains.base import Memory
from langchain.chains.conversation.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
    SUMMARY_PROMPT,
)
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate


class ConversationBufferMemory(Memory, BaseModel):
    """Buffer for storing conversation memory."""

    buffer: str = ""
    dynamic_key: str = "history"  #: :meta private:

    @property
    def dynamic_keys(self) -> List[str]:
        """Will always return list of dynamic keys.

        :meta private:
        """
        return [self.dynamic_key]

    def _load_dynamic_keys(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.dynamic_key: self.buffer}

    def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        prompt_input_keys = list(set(inputs.keys()) - set(self.dynamic_keys))
        assert len(prompt_input_keys) == 1, "One input key expected"
        assert len(outputs.keys()) == 1, "One output key expected"
        human = "Human: " + inputs[prompt_input_keys[0]]
        ai = "AI: " + outputs[list(outputs.keys())[0]]
        self.buffer += "\n" + "\n".join([human, ai])


class ConversationEntityMemory(Memory, BaseModel):
    """Conversation entity search and store to memory."""

    buffer: List[dict] = Field(default_factory=list)
    history_window_len: int = 5
    docstore: InMemoryDocstore
    llm: LLM
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT
    dynamic_keys: List[str] = ["history", "entities"]  #: :meta private:
    entities_per_turn: List[str] = Field(default_factory=list)  #: :meta private:

    def _format_buffer_to_string(self) -> str:
        section = self.buffer[: -self.history_window_len]
        result = []
        for line in section:
            result.append("Human: {input}".format(input=line["input"]))
            result.append("AI: {response}".format(response=line["response"]))
        return "\n".join(result)

    def _load_dynamic_keys(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer with entity summaries."""
        prompt_input_keys = list(set(inputs.keys()) - set(self.dynamic_keys))
        assert len(prompt_input_keys) == 1, "One input key expected"
        input = inputs[prompt_input_keys[0]]
        full_conversation = self._format_buffer_to_string() + "\nHuman: " + input
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)
        entities = chain.predict(history=full_conversation)
        if not entities:
            self.entities_per_turn = []
            all_results = ""
        else:
            all_results = ""
            all_entities = []
            for entity in entities.split(","):
                entity = entity.strip()
                all_entities.append(entity)
                entity_summary = self.docstore.search(entity)
                if isinstance(entity_summary, str):
                    continue
                all_results += "\nHere's what we know about {entity}".format(
                    entity=entity
                )
                all_results += "\n" + entity_summary.page_content
            self.entities_per_turn = all_entities
        return {"entities": all_results, "history": self._format_buffer_to_string()}

    def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        prompt_input_keys = list(set(inputs.keys()) - set(self.dynamic_keys))
        assert len(prompt_input_keys) == 1, "One input key expected"
        assert len(outputs.keys()) == 1, "One output key expected"
        assert self.entity_summarization_prompt.input_variables == [
            "history",
            "entity",
            "summary",
        ], "Expected different input keys"
        human = inputs[prompt_input_keys[0]]
        ai = outputs[list(outputs.keys())[0]]
        full_conversation = self._format_buffer_to_string() + "\nHuman: " + human
        for entity in self.entities_per_turn:
            chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)
            entity_summary = self.docstore.search(entity) or ""
            summary = chain.predict(
                input=human,
                entity=entity,
                summary=entity_summary,
                history=full_conversation,
            )
            self.docstore.add({entity: Document(page_content=summary)})
        self.buffer.append({"input": human, "response": ai})


class ConversationSummaryMemory(Memory, BaseModel):
    """Conversation summarizer to memory."""

    buffer: str = ""
    llm: LLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    dynamic_key: str = "history"  #: :meta private:

    @property
    def dynamic_keys(self) -> List[str]:
        """Will always return list of dynamic keys.

        :meta private:
        """
        return [self.dynamic_key]

    def _load_dynamic_keys(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.dynamic_key: self.buffer}

    def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        prompt_input_keys = list(set(inputs.keys()) - set(self.dynamic_keys))
        assert len(prompt_input_keys) == 1, "One input key expected"
        assert len(outputs.keys()) == 1, "One output key expected"
        assert self.prompt.input_variables == [
            "summary",
            "new_lines",
        ], "Expected different input keys"
        human = "Human: " + inputs[prompt_input_keys[0]]
        ai = "AI: " + outputs[list(outputs.keys())[0]]
        new_lines = "\n".join([human, ai])
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.buffer = chain.predict(summary=self.buffer, new_lines=new_lines)
