"""Router-Expert framework."""
from typing import Callable, Dict, List, NamedTuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.input import ChainedInput, get_color_mapping
from langchain.routing_chains.router import Router


class ToolConfig(NamedTuple):
    """Configuration for tools."""

    tool_name: str
    tool: Callable[[str], str]


class RoutingChain(Chain, BaseModel):
    """Chain that uses a router to use tools."""

    router: Router
    """Router to use."""
    tool_configs: List[ToolConfig]
    """Tool configs this chain has access to."""
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
        name_to_tool_map = {tc.tool_name: tc.tool for tc in self.tool_configs}
        starter_string = (
            inputs[self.input_key]
            + self.router.starter_string
            + self.router.router_prefix
        )
        chained_input = ChainedInput(starter_string, verbose=self.verbose)
        color_mapping = get_color_mapping(
            [c.tool_name for c in self.tool_configs], excluded_colors=["green"]
        )
        while True:
            output = self.router.route(chained_input.input)
            chained_input.add(output.log, color="green")
            if output.tool == self.router.finish_tool_name:
                return {self.output_key: output.tool_input}
            chain = name_to_tool_map[output.tool]
            observation = chain(output.tool_input)
            chained_input.add(f"\n{self.router.observation_prefix}")
            chained_input.add(observation, color=color_mapping[output.tool])
            chained_input.add(f"\n{self.router.router_prefix}")
