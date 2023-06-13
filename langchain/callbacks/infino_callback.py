"""Callback Handler that logs to infino."""
import datetime as dt
import time
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


def import_infino() -> Any:
    try:
        from infinopy import InfinoClient
    except ImportError:
        raise ImportError(
            "To use the Infino callbacks manager you need to have the"
            " `infino` python package installed."
            "Please install it with from Infino's repo."
        )
    return InfinoClient()


class InfinoCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to infino."""

    def __init__(self) -> None:
        # Set infino client
        self.client = import_infino()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Log the prompts to infino."""
        self.tokens_stream = []
        for prompt in prompts:
            payload = {"date": int(time.time()), "prompt_question": prompt}
            self.client.append_log(payload)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Need to register new token"""
        self.tokens_stream += token

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log the response to infino."""
        payload = {
            "date": int(time.time()),
            "prompt_response": self.tokens_stream.join(" "),
        }
        self.client.append_log(payload)
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Need to log the error."""
        payload = {
            "date": int(time.time()),
            "error": error,
            "labels": {"model": self.chain_name},
        }
        self.client.append_ts(payload)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Need to store the chain name."""
        class_name = serialized["name"]
        self.chain_name = class_name

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Need to log the error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing."""
        pass
