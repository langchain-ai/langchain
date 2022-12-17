"""Chain that takes in an input and produces an action and action input."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, root_validator

from langchain.agents.input import ChainedInput
from langchain.agents.tools import Tool
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.input import get_color_mapping
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import AgentAction


class Agent(Chain, BaseModel, ABC):
    """Agent that uses an LLM."""

    prompt: ClassVar[BasePromptTemplate]
    llm_chain: LLMChain
    tools: List[Tool]
    return_intermediate_steps: bool = False
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return [self.output_key, "intermediate_steps"]
        else:
            return [self.output_key]

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """Validate that prompt matches format."""
        prompt = values["llm_chain"].prompt
        if "agent_scratchpad" not in prompt.input_variables:
            raise ValueError(
                "`agent_scratchpad` should be a variable in prompt.input_variables"
            )
        return values

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool to use to finish the chain."""
        return "Final Answer"

    @property
    def starter_string(self) -> str:
        """Put this string after user input but before first LLM call."""
        return "\n"

    @abstractmethod
    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract tool and tool input from llm output."""

    def _fix_text(self, text: str) -> str:
        """Fix the text."""
        raise ValueError("fix_text not implemented for this agent.")

    @property
    def _stop(self) -> List[str]:
        return [f"\n{self.observation_prefix}"]

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        """Validate that appropriate tools are passed in."""
        pass

    @classmethod
    def create_prompt(cls, tools: List[Tool]) -> BasePromptTemplate:
        """Create a prompt for this class."""
        return cls.prompt

    def _prepare_for_new_call(self) -> None:
        pass

    @classmethod
    def from_llm_and_tools(cls, llm: LLM, tools: List[Tool], **kwargs: Any) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        llm_chain = LLMChain(llm=llm, prompt=cls.create_prompt(tools))
        return cls(llm_chain=llm_chain, tools=tools, **kwargs)

    def get_action(self, thoughts: str, inputs: dict) -> AgentAction:
        """Given input, decided what to do.

        Args:
            thoughts: LLM thoughts
            inputs: user inputs

        Returns:
            Action specifying what tool to use.
        """
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**inputs, **new_inputs}
        full_output = self.llm_chain.predict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            output = self.llm_chain.predict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        tool, tool_input = parsed_output
        return AgentAction(tool, tool_input, full_output)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Do any preparation necessary when receiving a new input.
        self._prepare_for_new_call()
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool.func for tool in self.tools}
        # We use the ChainedInput class to iteratively add to the input over time.
        chained_input = ChainedInput(self.llm_prefix, verbose=self.verbose)
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        # We now enter the agent loop (until it returns something).
        while True:
            # Call the LLM to see what to do.
            output = self.get_action(chained_input.input, inputs)
            # If the tool chosen is the finishing tool, then we end and return.
            if output.tool == self.finish_tool_name:
                final_output: dict = {self.output_key: output.tool_input}
                if self.return_intermediate_steps:
                    final_output[
                        "intermediate_steps"
                    ] = chained_input.intermediate_steps
                return final_output
            # Other we add the log to the Chained Input.
            chained_input.add_action(output, color="green")
            # And then we lookup the tool
            if output.tool in name_to_tool_map:
                chain = name_to_tool_map[output.tool]
                # We then call the tool on the tool input to get an observation
                observation = chain(output.tool_input)
                color = color_mapping[output.tool]
            else:
                observation = f"{output.tool} is not a valid tool, try another one."
                color = None
            # We then log the observation
            chained_input.add_observation(
                observation,
                self.observation_prefix,
                self.llm_prefix,
                color=color,
            )
