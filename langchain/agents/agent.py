"""Chain that takes in an input and produces an action and action input."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Tuple

from pydantic import BaseModel

from langchain.agents.tools import Tool
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.input import ChainedInput, get_color_mapping
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate


class Action(NamedTuple):
    """Action to take."""

    tool: str
    tool_input: str
    log: str


class Agent(Chain, BaseModel, ABC):
    """Agent that uses an LLM."""

    prompt: ClassVar[BasePromptTemplate]
    llm_chain: LLMChain
    tools: List[Tool]
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

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

    def get_action(self, text: str) -> Action:
        """Given input, decided what to do.

        Args:
            text: input string

        Returns:
            Action specifying what tool to use.
        """
        input_key = self.llm_chain.input_keys[0]
        inputs = {input_key: text, "stop": self._stop}
        full_output = self.llm_chain.predict(**inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            inputs = {input_key: text + full_output, "stop": self._stop}
            output = self.llm_chain.predict(**inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        tool, tool_input = parsed_output
        return Action(tool, tool_input, full_output)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run text through and get agent response."""
        text = inputs[self.input_key]
        # Do any preparation necessary when receiving a new input.
        self._prepare_for_new_call()
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool.func for tool in self.tools}
        # Construct the initial string to pass into the LLM. This is made up
        # of the user input, the special starter string, and then the LLM prefix.
        # The starter string is a special string that may be used by a LLM to
        # immediately follow the user input. The LLM prefix is a string that
        # prompts the LLM to take an action.
        starter_string = text + self.starter_string + self.llm_prefix
        # We use the ChainedInput class to iteratively add to the input over time.
        chained_input = ChainedInput(starter_string, verbose=self.verbose)
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        # We now enter the agent loop (until it returns something).
        while True:
            # Call the LLM to see what to do.
            output = self.get_action(chained_input.input)
            # Add the log to the Chained Input.
            chained_input.add(output.log, color="green")
            # If the tool chosen is the finishing tool, then we end and return.
            if output.tool == self.finish_tool_name:
                return {self.output_key: output.tool_input}
            # Otherwise we lookup the tool
            chain = name_to_tool_map[output.tool]
            # We then call the tool on the tool input to get an observation
            observation = chain(output.tool_input)
            # We then log the observation
            chained_input.add(f"\n{self.observation_prefix}")
            chained_input.add(observation, color=color_mapping[output.tool])
            # We then add the LLM prefix into the prompt to get the LLM to start
            # thinking, and start the loop all over.
            chained_input.add(f"\n{self.llm_prefix}")
