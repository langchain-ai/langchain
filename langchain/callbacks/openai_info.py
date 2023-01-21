"""Callback Handler that prints to std out."""
from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish, LLMResult


class OpenAICallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks OpenAI info."""
    total_tokens: int = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_end(self, response: LLMResult) -> None:
        """Do nothing."""
        if "token_usage" in response.llm_output:
            if "total_tokens" in response.llm_output["token_usage"]:
                self.total_tokens += response.llm_output["token_usage"]["total_tokens"]

    def on_llm_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Print out that we finished a chain."""
        pass

    def on_chain_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        action: AgentAction,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        pass

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        pass

    def on_tool_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        """Run when agent ends."""
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        pass
