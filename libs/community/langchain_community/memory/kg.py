from typing import Any, Dict, List, Type, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from pydantic import Field

from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.networkx_graph import (
    KnowledgeTriple,
    get_entities,
    parse_triples,
)

try:
    from langchain.chains.llm import LLMChain
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.memory.prompt import (
        ENTITY_EXTRACTION_PROMPT,
        KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
    )
    from langchain.memory.utils import get_prompt_input_key

    class ConversationKGMemory(BaseChatMemory):
        """Knowledge graph conversation memory.

        Integrates with external knowledge graph to store and retrieve
        information about knowledge triples in the conversation.
        """

        k: int = 2
        human_prefix: str = "Human"
        ai_prefix: str = "AI"
        kg: NetworkxEntityGraph = Field(default_factory=NetworkxEntityGraph)
        knowledge_extraction_prompt: BasePromptTemplate = (
            KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
        )
        entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
        llm: BaseLanguageModel
        summary_message_cls: Type[BaseMessage] = SystemMessage
        """Number of previous utterances to include in the context."""
        memory_key: str = "history"  #: :meta private:

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Return history buffer."""
            entities = self._get_current_entities(inputs)

            summary_strings = []
            for entity in entities:
                knowledge = self.kg.get_entity_knowledge(entity)
                if knowledge:
                    summary = f"On {entity}: {'. '.join(knowledge)}."
                    summary_strings.append(summary)
            context: Union[str, List]
            if not summary_strings:
                context = [] if self.return_messages else ""
            elif self.return_messages:
                context = [
                    self.summary_message_cls(content=text) for text in summary_strings
                ]
            else:
                context = "\n".join(summary_strings)

            return {self.memory_key: context}

        @property
        def memory_variables(self) -> List[str]:
            """Will always return list of memory variables.

            :meta private:
            """
            return [self.memory_key]

        def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
            """Get the input key for the prompt."""
            if self.input_key is None:
                return get_prompt_input_key(inputs, self.memory_variables)
            return self.input_key

        def _get_prompt_output_key(self, outputs: Dict[str, Any]) -> str:
            """Get the output key for the prompt."""
            if self.output_key is None:
                if len(outputs) != 1:
                    raise ValueError(f"One output key expected, got {outputs.keys()}")
                return list(outputs.keys())[0]
            return self.output_key

        def get_current_entities(self, input_string: str) -> List[str]:
            chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)
            buffer_string = get_buffer_string(
                self.chat_memory.messages[-self.k * 2 :],
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
            output = chain.predict(
                history=buffer_string,
                input=input_string,
            )
            return get_entities(output)

        def _get_current_entities(self, inputs: Dict[str, Any]) -> List[str]:
            """Get the current entities in the conversation."""
            prompt_input_key = self._get_prompt_input_key(inputs)
            return self.get_current_entities(inputs[prompt_input_key])

        def get_knowledge_triplets(self, input_string: str) -> List[KnowledgeTriple]:
            chain = LLMChain(llm=self.llm, prompt=self.knowledge_extraction_prompt)
            buffer_string = get_buffer_string(
                self.chat_memory.messages[-self.k * 2 :],
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
            output = chain.predict(
                history=buffer_string,
                input=input_string,
                verbose=True,
            )
            knowledge = parse_triples(output)
            return knowledge

        def _get_and_update_kg(self, inputs: Dict[str, Any]) -> None:
            """Get and update knowledge graph from the conversation history."""
            prompt_input_key = self._get_prompt_input_key(inputs)
            knowledge = self.get_knowledge_triplets(inputs[prompt_input_key])
            for triple in knowledge:
                self.kg.add_triple(triple)

        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
            """Save context from this conversation to buffer."""
            super().save_context(inputs, outputs)
            self._get_and_update_kg(inputs)

        def clear(self) -> None:
            """Clear memory contents."""
            super().clear()
            self.kg.clear()

except ImportError:
    # Placeholder object
    class ConversationKGMemory:  # type: ignore[no-redef]
        pass
