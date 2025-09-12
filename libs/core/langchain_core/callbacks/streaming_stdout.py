"""Callback Handler streams to stdout on new llm token."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain_core.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running.

        Args:
            serialized (dict[str, Any]): The serialized LLM.
            prompts (list[str]): The prompts to run.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running.

        Args:
            serialized (dict[str, Any]): The serialized LLM.
            messages (list[list[BaseMessage]]): The messages to run.
            **kwargs (Any): Additional keyword arguments.
        """

    @override
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token (str): The new token.
            **kwargs (Any): Additional keyword arguments.
        """
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response from the LLM.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors.

        Args:
            error (BaseException): The error that occurred.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when a chain starts running.

        Args:
            serialized (dict[str, Any]): The serialized chain.
            inputs (dict[str, Any]): The inputs to the chain.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Run when a chain ends running.

        Args:
            outputs (dict[str, Any]): The outputs of the chain.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors.

        Args:
            error (BaseException): The error that occurred.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when the tool starts running.

        Args:
            serialized (dict[str, Any]): The serialized tool.
            input_str (str): The input string.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action.

        Args:
            action (AgentAction): The agent action.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running.

        Args:
            output (Any): The output of the tool.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors.

        Args:
            error (BaseException): The error that occurred.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on an arbitrary text.

        Args:
            text (str): The text to print.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on the agent end.

        Args:
            finish (AgentFinish): The agent finish.
            **kwargs (Any): Additional keyword arguments.
        """
