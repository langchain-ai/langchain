from typing import Any, Optional

from langchain.input import print_text
from langchain.schema import AgentAction


class BaseLogger:
    def log_agent_start(self, text: str, **kwargs: Any):
        pass

    def log_agent_end(self, text: str, **kwargs: Any):
        pass

    def log_agent_action(self, action: AgentAction, **kwargs: Any):
        pass

    def log_agent_observation(self, observation: str, **kwargs: Any):
        pass

    def log_llm_inputs(self, inputs: dict, prompt: str, **kwargs):
        pass

    def log_llm_response(self, output: str, **kwargs):
        pass


class StOutLogger(BaseLogger):
    def log_agent_start(self, text: str, **kwargs: Any):
        print_text(text)

    def log_agent_end(self, text: str, **kwargs: Any):
        pass

    def log_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ):
        print_text(action.log, color=color)

    def log_llm_inputs(self, inputs: dict, prompt: str, **kwargs):
        print("Prompt after formatting:")
        print_text(prompt, color="green", end="\n")

    def log_llm_response(self, output: str, **kwargs):
        pass

    def log_agent_observation(
        self,
        observation: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ):
        print_text(f"\n{observation_prefix}")
        print_text(observation, color=color)
        print_text(f"\n{llm_prefix}")
