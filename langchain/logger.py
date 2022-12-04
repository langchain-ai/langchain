from langchain.schema import AgentAction
from typing import Optional, Any
from langchain.input import print_text
import logging
logging.basicConfig


class BaseLogger:

    def log_agent_start(self, text: str, **kwargs: Any):
        pass

    def log_agent_end(self, text: str, **kwargs: Any):
        pass

    def log_agent_action(self, action: AgentAction, **kwargs: Any):
        pass

    def log_agent_observation(self, observation: str, **kwargs: Any):
        pass


class StOutLogger(BaseLogger):
    def log_agent_start(self, text: str, **kwargs: Any):
        print_text(text)

    def log_agent_end(self, text: str, **kwargs: Any):
        pass

    def log_agent_action(self, action: AgentAction, color: Optional[str] = None, **kwargs: Any):
        print_text(action.log, color=color)

    def log_agent_observation(
            self,
            observation: str,
            color: Optional[str] = None,
            observation_prefix: Optional[str] = None,
            llm_prefix: Optional[str] = None,
            **kwargs: Any):
        print_text(f"\n{observation_prefix}")
        print_text(observation, color=color)
        print_text(f"\n{llm_prefix}")



logger = StOutLogger()