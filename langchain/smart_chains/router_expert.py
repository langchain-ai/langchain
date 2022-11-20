"""Router-Expert framework."""
from typing import Callable, Dict, List, NamedTuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.input import ChainedInput, get_color_mapping
from langchain.smart_chains.router import RouterChain


class ExpertConfig(NamedTuple):
    """Configuration for experts."""

    expert_name: str
    expert: Callable[[str], str]


class RouterExpertChain(Chain, BaseModel):
    """Chain that implements the Router/Expert system."""

    router_chain: RouterChain
    """Router chain."""
    expert_configs: List[ExpertConfig]
    """Expert configs this chain has access to."""
    starter_string: str = "\n"
    """String to put after user input but before first router."""
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
        action_to_chain_map = {e.expert_name: e.expert for e in self.expert_configs}
        starter_string = (
            inputs[self.input_key]
            + self.starter_string
            + self.router_chain.router_prefix
        )
        chained_input = ChainedInput(
            starter_string,
            verbose=self.verbose,
        )
        color_mapping = get_color_mapping(
            [c.expert_name for c in self.expert_configs], excluded_colors=["green"]
        )
        while True:
            action, action_input, log = self.router_chain.get_action_and_input(
                chained_input.input
            )
            chained_input.add(log, color="green")
            if action == self.router_chain.finish_action_name:
                return {self.output_key: action_input}
            chain = action_to_chain_map[action]
            ca = chain(action_input)
            chained_input.add(f"\n{self.router_chain.observation_prefix}")
            chained_input.add(ca, color=color_mapping[action])
            chained_input.add(f"\n{self.router_chain.router_prefix}")
