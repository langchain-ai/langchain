"""Chain that takes in an input and produces an action and action input."""
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Tuple

from pydantic import BaseModel

from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.routing_chains.tools import Tool


class RouterOutput(NamedTuple):
    """Output of a router."""

    tool: str
    tool_input: str
    log: str


class Router(ABC):
    """Chain responsible for deciding the action to take."""

    @abstractmethod
    def route(self, text: str) -> RouterOutput:
        """Given input, decided how to route it.

        Args:
            text: input string

        Returns:
            RouterOutput specifying what tool to use.
        """

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""

    @property
    @abstractmethod
    def router_prefix(self) -> str:
        """Prefix to append the router call with."""

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool to use to finish the chain."""
        return "Final Answer"

    @property
    def starter_string(self) -> str:
        """Put this string after user input but before first router call."""
        return "\n"


class LLMRouter(Router, BaseModel, ABC):
    """Router that uses an LLM."""

    llm_chain: LLMChain

    @abstractmethod
    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract tool and tool input from llm output."""

    def _fix_text(self, text: str) -> str:
        """Fix the text."""
        raise ValueError("fix_text not implemented for this router.")

    @property
    def _stop(self) -> List[str]:
        return [f"\n{self.observation_prefix}"]

    @classmethod
    @abstractmethod
    def from_llm_and_tools(cls, llm: LLM, tools: List[Tool]) -> "Router":
        """Construct a router from an LLM and tools."""

    def route(self, text: str) -> RouterOutput:
        """Given input, decided how to route it.

        Args:
            text: input string

        Returns:
            RouterOutput specifying what tool to use.
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
        return RouterOutput(tool, tool_input, full_output)
