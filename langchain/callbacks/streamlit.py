"""Callback Handler that logs to streamlit."""
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to streamlit."""

    def __init__(self) -> None:
        try:
            import streamlit as st
        except ImportError as e:
            raise ImportError(
                "Could not import streamlit Python package. "
                "Please install it with `pip install streamlit`."
            ) from e

        self.tokens_area = st.empty()
        self.tokens_stream = ""
        self.st = st

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        self.st.write("Prompts after formatting:")
        for prompt in prompts:
            self.st.write(prompt)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.write(self.tokens_stream)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        self.st.write(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        self.st.write("Finished chain.")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
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
        """Run on agent action."""
        # st.write requires two spaces before a newline to render it

        self.st.markdown(action.log.replace("\n", "  \n"))

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        self.st.write(f"{observation_prefix}{output}")
        self.st.write(llm_prefix)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on text."""
        # st.write requires two spaces before a newline to render it
        self.st.write(text.replace("\n", "  \n"))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        # st.write requires two spaces before a newline to render it
        self.st.write(finish.log.replace("\n", "  \n"))
