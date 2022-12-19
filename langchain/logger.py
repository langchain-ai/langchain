"""BETA: everything in here is highly experimental, do not rely on."""
from typing import Any, Optional

from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish


class BaseLogger:
    """Base logging interface."""

    def log_agent_start(self, text: str, **kwargs: Any) -> None:
        """Log the start of an agent interaction."""
        pass

    def log_agent_end(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Log the end of an agent interaction."""
        pass

    def log_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Log agent action decision."""
        pass

    def log_agent_observation(self, observation: str, **kwargs: Any) -> None:
        """Log agent observation."""
        pass

    def log_llm_inputs(self, inputs: dict, prompt: str, **kwargs: Any) -> None:
        """Log LLM inputs."""
        pass

    def log_llm_response(self, output: str, **kwargs: Any) -> None:
        """Log LLM response."""
        pass


class StdOutLogger(BaseLogger):
    """Interface for printing things to stdout."""

    def log_agent_start(self, text: str, **kwargs: Any) -> None:
        """Print the text to start the agent."""
        print_text(text)

    def log_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Print the log of the action in a certain color."""
        print_text(action.log, color=color)

    def log_agent_observation(
        self,
        observation: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Print the observation in a special color."""
        print_text(f"\n{observation_prefix}")
        print_text(observation, color=color)
        print_text(f"\n{llm_prefix}")

    def log_llm_inputs(self, inputs: dict, prompt: str, **kwargs: Any) -> None:
        """Print the prompt in green."""
        print("Prompt after formatting:")
        print_text(prompt, color="green", end="\n")

    def log_agent_end(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Log the end of an agent interaction."""
        print_text(finish.log, color=color)
