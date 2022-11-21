"""Router-Expert framework."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.input import ChainedInput, get_color_mapping
from langchain.routing_chains.router import Router
from langchain.routing_chains.tools import Tool


class RoutingChain(Chain, BaseModel):
    """Chain that uses a router to use tools."""

    router: Router
    """Router to use."""
    tools: List[Tool]
    """Tools this chain has access to."""
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
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool.func for tool in self.tools}
        # Construct the initial string to pass into the router. This is made up
        # of the user input, the special starter string, and then the router prefix.
        # The starter string is a special string that may be used by a router to
        # immediately follow the user input. The router prefix is a string that
        # prompts the router to start routing.
        starter_string = (
            inputs[self.input_key]
            + self.router.starter_string
            + self.router.router_prefix
        )
        # We use the ChainedInput class to iteratively add to the input over time.
        chained_input = ChainedInput(starter_string, verbose=self.verbose)
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        # We now enter the router loop (until it returns something).
        while True:
            # Call the router to see what to do.
            output = self.router.route(chained_input.input)
            # Add the log to the Chained Input.
            chained_input.add(output.log, color="green")
            # If the tool chosen is the finishing tool, then we end and return.
            if output.tool == self.router.finish_tool_name:
                return {self.output_key: output.tool_input}
            # Otherwise we lookup the tool
            chain = name_to_tool_map[output.tool]
            # We then call the tool on the tool input to get an observation
            observation = chain(output.tool_input)
            # We then log the observation
            chained_input.add(f"\n{self.router.observation_prefix}")
            chained_input.add(observation, color=color_mapping[output.tool])
            # We then add the router prefix into the prompt to get the router to start
            # thinking, and start the loop all over.
            chained_input.add(f"\n{self.router.router_prefix}")
