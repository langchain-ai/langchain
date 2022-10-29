from langchain.chains.base import Chain
from langchain.llms.base import LLM
from langchain.chains.llm import LLMChain
from langchain.prompt import Prompt
from pydantic import BaseModel, Extra
from typing import List, Dict, Callable, NamedTuple
from langchain.chains.jurassic_x.prompt import BASE_TEMPLATE


class ChainConfig(NamedTuple):
    action_name: str
    action: Callable
    action_description: str



class JurassicXChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.

    Example:
        .. code-block:: python

            from langchain import LLMMathChain, OpenAI
            llm_math = LLMMathChain(llm=OpenAI())
    """

    llm: LLM
    """LLM wrapper to use."""
    prompt: Prompt
    action_to_chain_map: Dict[str, Callable]
    verbose: bool = False
    """Whether to print out the code that was executed."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    @classmethod
    def from_chains(cls, llm: LLM, chains: ChainConfig, **kwargs) -> 'JurassicXChain':
        tools = "\n".join([f"{chain.action_name}: {chain.action_description}" for chain in chains])
        tool_names = ', '.join([chain.action_name for chain in chains])
        template = BASE_TEMPLATE.format(tools=tools, tool_names=tool_names)
        prompt = Prompt(template=template, input_variables=["input"])
        action_to_chain_map = {chain.action_name: chain.action for chain in chains}
        return cls(llm=llm, prompt=prompt, action_to_chain_map=action_to_chain_map, **kwargs)


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
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _run(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        _input = f"{inputs[self.input_key]}\nThought:"
        while True:
            observation = llm_chain.predict(input=_input, stop=["\nObservation"])
            print(observation)
            ps = [p for p in observation.split('\n') if p]
            if ps[-1].startswith("Final Answer"):
                return {self.output_key: ps[-1][len('Final Answer: '):]}
            assert ps[-1].startswith('Action Input: ')
            assert ps[-2].startswith('Action: ')
            action = ps[-2][len('Action: '):]
            action_input = ps[-1][len('Action Input: '):]
            chain = self.action_to_chain_map[action]
            ca = chain(action_input)
            print(ca)
            _input = _input + observation + f"\nObservation: {ca}\nThought:"

    def run(self, question: str) -> str:
        return self({self.input_key: question})[self.output_key]
