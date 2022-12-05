"""Input manager for agents."""
from typing import Optional

import langchain
from langchain.schema import AgentAction


class ChainedInput:
    """Class for working with input that is the result of chains."""

    def __init__(self, text: str, verbose: bool = False):
        """Initialize with verbose flag and initial text."""
        self._verbose = verbose
        if self._verbose:
            langchain.logger.log_agent_start(text)
        self._input = text

    def add_action(self, action: AgentAction, color: Optional[str] = None) -> None:
        """Add text to input, print if in verbose mode."""
        if self._verbose:
            langchain.logger.log_agent_action(action, color=color)
        self._input += action.log

    def add_observation(
        self,
        observation: str,
        observation_prefix: str,
        llm_prefix: str,
        color: Optional[str],
    ) -> None:
        """Add observation to input, print if in verbose mode."""
        if self._verbose:
            langchain.logger.log_agent_observation(
                observation,
                color=color,
                observation_prefix=observation_prefix,
                llm_prefix=llm_prefix,
            )
        self._input += f"\n{observation_prefix}{observation}\n{llm_prefix}"

    @property
    def input(self) -> str:
        """Return the accumulated input."""
        return self._input
