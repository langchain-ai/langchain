"""Input manager for agents."""
from typing import Optional

from langchain.schema import AgentAction
from langchain.logger import logger


class ChainedInput:
    """Class for working with input that is the result of chains."""

    def __init__(self, text: str, observation_prefix: str, llm_prefix: str, verbose: bool = False):
        """Initialize with verbose flag and initial text."""
        self._verbose = verbose
        if self._verbose:
            logger.log_agent_start(text)
        self._input = text
        self._observation_prefix = observation_prefix
        self._llm_prefix = llm_prefix

    def add_action(self, action: AgentAction, color: Optional[str] = None) -> None:
        """Add text to input, print if in verbose mode."""
        if self._verbose:
            logger.log_agent_action(action, color=color)
        self._input += action.log

    def add_observation(self, observation: str, color: Optional[str]) -> None:
        """Add observation to input, print if in verbose mode."""
        if self._verbose:
            logger.log_agent_observation(
                observation,
                color=color,
                observation_prefix=self._observation_prefix,
                llm_prefix=self._llm_prefix,
            )
        self._input += f"\n{self._observation_prefix}{observation}\n{self._llm_prefix}"

    @property
    def input(self) -> str:
        """Return the accumulated input."""
        return self._input
