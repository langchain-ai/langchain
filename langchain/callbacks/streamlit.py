"""Callback Handler that logs to streamlit."""
from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.input import print_text
from langchain.schema import AgentAction, LLMResult
import streamlit as st


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to streamlit."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        st.write("Prompts after formatting:")
        for prompt in prompts:
            st.write(prompt)

    def on_llm_end(self, response: LLMResult) -> None:
        """Do nothing."""
        pass

    def on_llm_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        st.write(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Print out that we finished a chain."""
        st.write("\n\033[1m> Finished chain.\033[0m")

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
        st.write(action.log)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        st.write(f"\n{observation_prefix}")
        st.write(output)
        st.write(f"\n{llm_prefix}")

    def on_tool_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_agent_end(
        self, log: str, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run when agent ends."""
        st.write(log)
