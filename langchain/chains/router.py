"""Chain that takes in an input and produces an action and action input."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain


class RouterChain(Chain, BaseModel, ABC):
    """Chain responsible for deciding the action to take."""

    input_key: str = "input_text"  #: :meta private:
    action_key: str = "action"  #: :meta private:
    action_input_key: str = "action_input"  #: :meta private:
    log_key: str = "log"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Will be the input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return three keys: the action, the action input, and the log.

        :meta private:
        """
        return [self.action_key, self.action_input_key, self.log_key]

    @abstractmethod
    def get_action_and_input(self, text: str) -> Tuple[str, str, str]:
        """Return action, action input, and log (in that order)."""

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        action, action_input, log = self.get_action_and_input(inputs[self.input_key])
        return {
            self.action_key: action,
            self.action_input_key: action_input,
            self.log_key: log,
        }


class LLMRouterChain(RouterChain, BaseModel, ABC):
    """RouterChain that uses an LLM."""

    llm_chain: LLMChain
    stops: Optional[List[str]]

    @abstractmethod
    def _extract_action_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract action and action input from llm output."""

    def _fix_text(self, text: str) -> str:
        """Fix the text."""
        raise ValueError("fix_text not implemented for this router.")

    def get_action_and_input(self, text: str) -> Tuple[str, str, str]:
        """Return action, action input, and log (in that order)."""
        input_key = self.llm_chain.input_keys[0]
        inputs = {input_key: text, "stop": self.stops}
        full_output = self.llm_chain.predict(**inputs)
        parsed_output = self._extract_action_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            inputs = {input_key: text + full_output, "stop": self.stops}
            output = self.llm_chain.predict(**inputs)
            full_output += output
            parsed_output = self._extract_action_and_input(full_output)
        action, action_input = parsed_output
        return action, action_input, full_output
