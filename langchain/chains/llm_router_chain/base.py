import re
from types import FunctionType
from typing import Dict, List, Any
from pydantic import Extra, root_validator

from langchain import LLMChain, BasePromptTemplate
from langchain.chains.base import Chain
from langchain.chains.conversation.prompt import PROMPT
from langchain.input import get_color_mapping
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMemory


class RouterChain(LLMChain):
    """
    Router chain that picks the most relevant model to call based on vector queries.
    The chain includes support for memory
    """

    memory: BaseMemory = ConversationBufferWindowMemory(k=1)
    fuzzy_match_threshold = 1.5
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""
    last_chain: Chain = None
    chains: Dict[str, Chain]
    strip_outputs: bool = False
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:
    vector_lookup_fn: FunctionType = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "router_chain"

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        _input = inputs[self.input_key]
        last_chain_name = re.compile('Responding Chain - (.*?):').findall(inputs['history'])
        if last_chain_name and len(last_chain_name) > 0:
            self.last_chain = self.chains.get(last_chain_name[0])
        color_mapping = get_color_mapping([str(x) for x in self.chains.keys()])
        if not self.vector_lookup_fn:
            raise ValueError("Vector lookup function not provided for this router chain.")
        m_name, distance = self.vector_lookup_fn(query=[_input])
        # picking a guardrail where if the LLM response is way off - then just use the same model as the previous
        # one to continue conversing.
        if self.chains.get(m_name) and distance <= self.fuzzy_match_threshold:
            _input = self.chains[m_name](_input)
        else:
            if self.last_chain:
                m_name = last_chain_name[0]
                _input = self.last_chain(_input)
            else:
                raise ValueError(
                    "Suitable destination chain not found for this question. Distance computed from nearest match: " +
                    str(distance))
        self.callback_manager.on_text(
            str(_input['text']), color=color_mapping[m_name], end="\n", verbose=self.verbose
        )
        return {
            self.output_key: 'Responding Chain - ' + m_name + ':' + _input['text']
        }

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        memory_keys = values["memory"].memory_variables
        input_key = values["input_key"]
        if input_key in memory_keys:
            raise ValueError(
                f"The input key {input_key} was also found in the memory keys "
                f"({memory_keys}) - please provide keys that don't overlap."
            )
        prompt_variables = values["prompt"].input_variables
        expected_keys = memory_keys + [input_key]
        if set(expected_keys) != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but got {memory_keys} as inputs from "
                f"memory, and {input_key} as the normal input key."
            )
        return values
