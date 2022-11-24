"""Memory modules for conversation prompts."""
from typing import Any, Dict, List

from pydantic import BaseModel, root_validator

from langchain.chains.base import Memory
from langchain.chains.conversation.prompt import SUMMARY_PROMPT
from langchain.chains.llm import LLMChain
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
        prompt_input_keys = list(set(inputs).difference(self.dynamic_keys))
        if len(prompt_input_keys) != 1:
            raise ValueError(f"One input key expected got {prompt_input_keys}")
        if len(outputs) != 1:
            raise ValueError(f"One output key expected, got {outputs.keys()}")
        human = "Human: " + inputs[prompt_input_keys[0]]
        ai = "AI: " + outputs[list(outputs.keys())[0]]
        self.buffer += "\n" + "\n".join([human, ai])


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

    def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        prompt_input_keys = list(set(inputs).difference(self.dynamic_keys))
        if len(prompt_input_keys) != 1:
            raise ValueError(f"One input key expected got {prompt_input_keys}")
        if len(outputs) != 1:
            raise ValueError(f"One output key expected, got {outputs.keys()}")
        human = "Human: " + inputs[prompt_input_keys[0]]
        ai = "AI: " + list(outputs.values())[0]
        new_lines = "\n".join([human, ai])
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.buffer = chain.predict(summary=self.buffer, new_lines=new_lines)
