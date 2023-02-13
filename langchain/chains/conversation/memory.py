"""Memory modules for conversation prompts."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator

from langchain.chains.base import Memory
from langchain.chains.conversation.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
    SUMMARY_PROMPT,
)
from langchain.chains.llm import LLMChain
from langchain.graphs.networkx_graph import (
    NetworkxEntityGraph,
    get_entities,
    parse_triples,
)
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


def _get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]


class CombinedMemory(Memory, BaseModel):
    """Class for combining multiple memories' data together."""

    memories: List[Memory]
    """For tracking all the memories that should be accessed."""

    @property
    def memory_variables(self) -> List[str]:
        """All the memory variables that this instance provides."""
        """Collected from the all the linked memories."""

        memory_variables = []

        for memory in self.memories:
            memory_variables.extend(memory.memory_variables)

        return memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load all vars from sub-memories."""
        memory_data: Dict[str, Any] = {}

        # Collect vars from all sub-memories
        for memory in self.memories:
            data = memory.load_memory_variables(inputs)
            memory_data = {
                **memory_data,
                **data,
            }

        return memory_data

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this session for every memory."""
        # Save context for all sub-memories
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear context from this session for every memory."""
        for memory in self.memories:
            memory.clear()


class ConversationBufferMemory(Memory, BaseModel):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    buffer: str = ""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "history"  #: :meta private:

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.buffer += "\n" + "\n".join([human, ai])

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = ""


class ConversationBufferWindowMemory(Memory, BaseModel):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    buffer: List[str] = Field(default_factory=list)
    memory_key: str = "history"  #: :meta private:
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    k: int = 5

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: "\n".join(self.buffer[-self.k :])}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.buffer.append("\n".join([human, ai]))

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = []


# For legacy naming reasons
ConversationalBufferWindowMemory = ConversationBufferWindowMemory


class ConversationSummaryMemory(Memory, BaseModel):
    """Conversation summarizer to memory."""

    buffer: str = ""
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    llm: BaseLLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    memory_key: str = "history"  #: :meta private:
    output_key: Optional[str] = None
    input_key: Optional[str] = None

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    @root_validator()
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
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: {inputs[prompt_input_key]}"
        ai = f"{self.ai_prefix}: {outputs[output_key]}"
        new_lines = "\n".join([human, ai])
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.buffer = chain.predict(summary=self.buffer, new_lines=new_lines)

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = ""


class ConversationEntityMemory(Memory, BaseModel):
    """Entity extractor & summarizer to memory."""

    buffer: List[str] = []
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    llm: BaseLLM
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    store: Dict[str, Optional[str]] = {}
    entity_cache: List[str] = []
    k: int = 3
    chat_history_key: str = "history"

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
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        output = chain.predict(
            history="\n".join(self.buffer[-self.k :]),
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
        return {
            self.chat_history_key: "\n".join(self.buffer[-self.k :]),
            "entities": entity_summaries,
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        for entity in self.entity_cache:
            chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)
            # key value store for entity
            existing_summary = self.store.get(entity, "")
            output = chain.predict(
                summary=existing_summary,
                history="\n".join(self.buffer[-self.k :]),
                input=inputs[prompt_input_key],
                entity=entity,
            )
            self.store[entity] = output.strip()
        new_lines = "\n".join([human, ai])
        self.buffer.append(new_lines)

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = []
        self.store = {}


class ConversationSummaryBufferMemory(Memory, BaseModel):
    """Buffer with summarizer for storing conversation memory."""

    buffer: List[str] = Field(default_factory=list)
    max_token_limit: int = 2000
    moving_summary_buffer: str = ""
    llm: BaseLLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    memory_key: str = "history"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    output_key: Optional[str] = None
    input_key: Optional[str] = None

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        if self.moving_summary_buffer == "":
            return {self.memory_key: "\n".join(self.buffer)}
        memory_val = self.moving_summary_buffer + "\n" + "\n".join(self.buffer)
        return {self.memory_key: memory_val}

    @root_validator()
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

    def get_num_tokens_list(self, arr: List[str]) -> List[int]:
        """Get list of number of tokens in each string in the input array."""
        return [self.llm.get_num_tokens(x) for x in arr]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: {inputs[prompt_input_key]}"
        ai = f"{self.ai_prefix}: {outputs[output_key]}"
        new_lines = "\n".join([human, ai])
        self.buffer.append(new_lines)
        # Prune buffer if it exceeds max token limit
        curr_buffer_length = sum(self.get_num_tokens_list(self.buffer))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(self.buffer.pop(0))
                curr_buffer_length = sum(self.get_num_tokens_list(self.buffer))
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            self.moving_summary_buffer = chain.predict(
                summary=self.moving_summary_buffer, new_lines=("\n".join(pruned_memory))
            )

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = []
        self.moving_summary_buffer = ""


class ConversationKGMemory(Memory, BaseModel):
    """Knowledge graph memory for storing conversation memory.

    Integrates with external knowledge graph to store and retrieve
    information about knowledge triples in the conversation.
    """

    k: int = 2
    buffer: List[str] = Field(default_factory=list)
    kg: NetworkxEntityGraph = Field(default_factory=NetworkxEntityGraph)
    knowledge_extraction_prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    llm: BaseLLM
    """Number of previous utterances to include in the context."""
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "history"  #: :meta private:

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        entities = self._get_current_entities(inputs)
        summaries = {}
        for entity in entities:
            knowledge = self.kg.get_entity_knowledge(entity)
            if knowledge:
                summaries[entity] = ". ".join(knowledge) + "."
        if summaries:
            summary_strings = [
                f"On {entity}: {summary}" for entity, summary in summaries.items()
            ]
            context_str = "\n".join(summary_strings)
        else:
            context_str = ""
        return {self.memory_key: context_str}

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return _get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def _get_prompt_output_key(self, outputs: Dict[str, Any]) -> str:
        """Get the output key for the prompt."""
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            return list(outputs.keys())[0]
        return self.output_key

    def _get_current_entities(self, inputs: Dict[str, Any]) -> List[str]:
        """Get the current entities in the conversation."""
        prompt_input_key = self._get_prompt_input_key(inputs)
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)
        output = chain.predict(
            history="\n".join(self.buffer[-self.k :]),
            input=inputs[prompt_input_key],
        )
        return get_entities(output)

    def _get_and_update_kg(self, inputs: Dict[str, Any]) -> None:
        """Get and update knowledge graph from the conversation history."""
        chain = LLMChain(llm=self.llm, prompt=self.knowledge_extraction_prompt)
        prompt_input_key = self._get_prompt_input_key(inputs)
        output = chain.predict(
            history="\n".join(self.buffer[-self.k :]),
            input=inputs[prompt_input_key],
            verbose=True,
        )
        knowledge = parse_triples(output)
        for triple in knowledge:
            self.kg.add_triple(triple)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        self._get_and_update_kg(inputs)
        prompt_input_key = self._get_prompt_input_key(inputs)
        output_key = self._get_prompt_output_key(outputs)
        human = f"{self.human_prefix}: {inputs[prompt_input_key]}"
        ai = f"{self.ai_prefix}: {outputs[output_key]}"
        new_lines = "\n".join([human.strip(), ai.strip()])
        self.buffer.append(new_lines)

    def clear(self) -> None:
        """Clear memory contents."""
        return self.kg.clear()
