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
            serialized: The serialized LLM.
            prompts: The prompts to run.
            **kwargs: Additional keyword arguments.
        """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running.

        Args:
            serialized: The serialized LLM.
            messages: The messages to run.
            **kwargs: Additional keyword arguments.
        """

    @override
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token: The new token.
            **kwargs: Additional keyword arguments.
        """
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response: The response from the LLM.
            **kwargs: Additional keyword arguments.
        """

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors.

        Args:
            error: The error that occurred.
            **kwargs: Additional keyword arguments.
        """

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when a chain starts running.

        Args:
            serialized: The serialized chain.
            inputs: The inputs to the chain.
            **kwargs: Additional keyword arguments.
        """

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Run when a chain ends running.

        Args:
            outputs: The outputs of the chain.
            **kwargs: Additional keyword arguments.
        """

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors.

        Args:
            error: The error that occurred.
            **kwargs: Additional keyword arguments.
        """

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when the tool starts running.

        Args:
            serialized: The serialized tool.
            input_str: The input string.
            **kwargs: Additional keyword arguments.
        """

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action.

        Args:
            action: The agent action.
            **kwargs: Additional keyword arguments.
        """

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running.

        Args:
            output: The output of the tool.
            **kwargs: Additional keyword arguments.
        """

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors.

        Args:
            error: The error that occurred.
            **kwargs: Additional keyword arguments.
        """

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on an arbitrary text.

        Args:
            text: The text to print.
            **kwargs: Additional keyword arguments.
        """

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on the agent end.

        Args:
            finish: The agent finish.
            **kwargs: Additional keyword arguments.
        """
