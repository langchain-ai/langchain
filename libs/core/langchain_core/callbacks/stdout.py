"""Callback Handler that prints to std out."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.utils import print_text

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


class StdOutCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler.

        Args:
            color: The color to use for the text. Defaults to None.
        """
        self.color = color

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Dict[str, Any]): The inputs to the chain.
            **kwargs (Any): Additional keyword arguments.
        """
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")  # noqa: T201

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain.

        Args:
            outputs (Dict[str, Any]): The outputs of the chain.
            **kwargs (Any): Additional keyword arguments.
        """
        print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action.

        Args:
            action (AgentAction): The agent action.
            color (Optional[str]): The color to use for the text. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
        print_text(action.log, color=color or self.color)

    def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation.

        Args:
            output (Any): The output to print.
            color (Optional[str]): The color to use for the text. Defaults to None.
            observation_prefix (Optional[str]): The observation prefix.
                Defaults to None.
            llm_prefix (Optional[str]): The LLM prefix. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
        output = str(output)
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color or self.color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when the agent ends.

        Args:
            text (str): The text to print.
            color (Optional[str]): The color to use for the text. Defaults to None.
            end (str): The end character to use. Defaults to "".
            **kwargs (Any): Additional keyword arguments.
        """
        print_text(text, color=color or self.color, end=end)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on the agent end.

        Args:
            finish (AgentFinish): The agent finish.
            color (Optional[str]): The color to use for the text. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
        print_text(finish.log, color=color or self.color, end="\n")
