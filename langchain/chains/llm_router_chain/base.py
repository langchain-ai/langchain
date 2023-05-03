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


class ConditionalRouterChain(Chain):
    """
    Router chain that picks the most relevant model to call based on a lookup function the caller would pass in.
    The function for example could be a vector query to do the determination of the destination
    The chain includes support for memory for maintaining a conversational context and leverages it for a fall back logic
    to defer subsequent messages to the same destination chain if the threshold for match is not met.
    """

    memory: BaseMemory = ConversationBufferWindowMemory(k=1)
    fuzzy_match_threshold = 1.5
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        pass

    last_chain: Chain = None
    chains: Dict[str, Chain]
    strip_outputs: bool = False
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:
    response_chain: str = "chain"
    responding_chain_hint: str = "dest_chain - "

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
        return [self.output_key, self.response_chain]

    @property
    def _chain_type(self) -> str:
        return "conditional_router_chain"

    def extract_dict_for_output_keys(self, val):
        if not val:
            raise ValueError(f'Empty value provided for output extraction. Got ${val}')
        result: Dict[str, str] = {}
        for k, v in val.items():
            if k in self.output_keys:
                result[k] = v
        return result

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        """Run the chain as text in, text out or multiple variables, text out."""
        if len(self.output_keys) != 2:
            raise ValueError(
                f"`run` not supported when there is not exactly "
                f"two output keys. Got {self.output_keys}."
            )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return self.extract_dict_for_output_keys(self(args[0]))

        if kwargs and not args:
            return self.extract_dict_for_output_keys(self(kwargs))

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )

    async def arun(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        return self.run(args, kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        _input = inputs[self.input_key]
        last_chain_name = re.compile('%s(.*?):' % self.responding_chain_hint).findall(inputs['history'])
        if last_chain_name and len(last_chain_name) > 0:
            self.last_chain = self.chains.get(last_chain_name[0])
        color_mapping = get_color_mapping([str(x) for x in self.chains.keys()])
        if not self.vector_lookup_fn:
            raise ValueError("Vector lookup function not provided for this router chain.")
        m_name, distance = self.vector_lookup_fn(query=[_input])
        # picking a guardrail where if the LLM response is way off - then just use the same model as the previous
        # one to continue conversing.
        if self.chains.get(m_name) and distance <= self.fuzzy_match_threshold:
            _output = self.chains[m_name](_input)
        else:
            if self.last_chain:
                m_name = last_chain_name[0]
                _output = self.last_chain(_input)
            else:
                raise ValueError(
                    "Suitable destination chain not found for this question. Distance computed from nearest match: " +
                    str(distance))
        self.callback_manager.on_text(
            str(_output['text']), color=color_mapping[m_name], end="\n", verbose=self.verbose
        )
        return {
            self.output_key: _output['text'],
            self.response_chain: m_name
        }

    def prep_outputs(
            self,
            inputs: Dict[str, str],
            outputs: Dict[str, str],
            return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if self.memory is not None:
            self.memory.save_context(inputs,
                                     {self.output_key: self.responding_chain_hint + outputs[self.response_chain] + ':' +
                                                       outputs[self.output_key]})
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}
