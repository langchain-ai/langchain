"""Memory modules for conversation prompts."""
from typing import Any, Dict, List

from pydantic import BaseModel, Field, root_validator

from langchain.chains.base import Memory
from langchain.chains.conversation.prompt import SUMMARY_PROMPT
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


def _get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]


class ConversationBufferMemory(Memory, BaseModel):
    """Buffer for storing conversation memory."""

    buffer: str = ""
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
        prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        if len(outputs) != 1:
            raise ValueError(f"One output key expected, got {outputs.keys()}")
        human = "Human: " + inputs[prompt_input_key]
        ai = "AI: " + outputs[list(outputs.keys())[0]]
        self.buffer += "\n" + "\n".join([human, ai])

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = ""


class ConversationalBufferWindowMemory(Memory, BaseModel):
    """Buffer for storing conversation memory."""

    buffer: List[str] = Field(default_factory=list)
    memory_key: str = "history"  #: :meta private:
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
        prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        if len(outputs) != 1:
            raise ValueError(f"One output key expected, got {outputs.keys()}")
        human = "Human: " + inputs[prompt_input_key]
        ai = "AI: " + outputs[list(outputs.keys())[0]]
        self.buffer.append("\n".join([human, ai]))

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = []


class ConversationSummaryMemory(Memory, BaseModel):
    """Conversation summarizer to memory."""

    buffer: str = ""
    llm: BaseLLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
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
        prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        if len(outputs) != 1:
            raise ValueError(f"One output key expected, got {outputs.keys()}")
        human = f"Human: {inputs[prompt_input_key]}"
        ai = f"AI: {list(outputs.values())[0]}"
        new_lines = "\n".join([human, ai])
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.buffer = chain.predict(summary=self.buffer, new_lines=new_lines)

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = ""

class ConversationSummaryBufferMemory(Memory, BaseModel):
    """Buffer with summarizer for storing conversation memory."""

    buffer: List[str] = Field(default_factory=list)
    max_token_limit: int = 256
    moving_summary_buffer: str = ""
    llm: BaseLLM
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    memory_key: str = "history"

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def to_json(self) -> Dict[str, str]:
        return {
            "buffer": self.buffer, 
            "moving_summary_buffer": self.moving_summary_buffer
        }

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        if self.moving_summary_buffer == "":
            return {self.memory_key: "\n".join(self.buffer)}
        return {self.memory_key: ("\n" + self.moving_summary_buffer + "\n" + "\n".join(self.buffer))}

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

    def get_num_tokens_list(self, arr: List[int]) -> List[int]:
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3 encoder instance
        enc = tiktoken.get_encoding("gpt2")

        # encode the list of text using the GPT-3 encoder
        tokenized_text = enc.encode_ordinary_batch(arr)

        # calculate the number of tokens for each encoded text in the list
        return [len(x) for x in tokenized_text]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        if len(outputs) != 1:
            raise ValueError(f"One output key expected, got {outputs.keys()}")
        human = f"Human: {inputs[prompt_input_key]}"
        ai = f"Assistant: {list(outputs.values())[0]}\n"
        new_lines = "\n".join([human, ai])
        self.buffer.append(new_lines)
        # Prune buffer if it exceeds max token limit
        curr_buffer_length = sum(self.get_num_tokens_list(self.buffer))
        while curr_buffer_length > self.max_token_limit:
            pruned_memory = self.buffer.pop(0)
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            self.moving_summary_buffer = chain.predict(summary=self.moving_summary_buffer, new_lines=("\n".join(pruned_memory)))
            curr_buffer_length = sum(self.get_num_tokens_list(self.buffer))

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = []
        self.moving_summary_buffer = ""
