"""Router-Expert framework."""
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.mrkl.prompt import BASE_TEMPLATE
from langchain.chains.router import LLMRouterChain
from langchain.input import ChainedInput, get_color_mapping
from langchain.llms.base import LLM
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.chains.router import RouterChain

FINAL_ANSWER_ACTION = "Final Answer: "


class ExpertConfig(NamedTuple):

    expert_name: str
    expert: Callable[[str], str]


class RouterExpertChain(Chain, BaseModel):
    """Chain that implements the Router/Expert system."""

    router_chain: RouterChain
    """Router chain."""
    expert_configs: List[ExpertConfig]
    """Expert configs this chain has access to."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        router_chain = MRKLRouterChain(self.llm, self.chain_configs)
        chained_input = ChainedInput(
            f"{inputs[self.input_key]}", verbose=self.verbose
        )
        color_mapping = get_color_mapping(
            [c.], excluded_colors=["green"]
        )
        while True:
            action, action_input, thought = router_chain.get_action_and_input(
                chained_input.input
            )
            chained_input.add(thought, color="green")
            if action == FINAL_ANSWER_ACTION:
                return {self.output_key: action_input}
            chain = self.action_to_chain_map[action]
            ca = chain(action_input)
            chained_input.add("\nObservation: ")
            chained_input.add(ca, color=color_mapping[action])
            chained_input.add("\nThought:")
