"""Callback Handler that prints to std out."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.utils import print_text

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


class StdOutCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self, color: str | None = None) -> None:
        """Initialize callback handler.

        Args:
            color: The color to use for the text.
        """
        self.color = color

    @override
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain.

        Args:
            serialized: The serialized chain.
            inputs: The inputs to the chain.
            **kwargs: Additional keyword arguments.
        """
        if "name" in kwargs:
            name = kwargs["name"]
        elif serialized:
            name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        else:
            name = "<unknown>"
        print(f"\n\n\033[1m> Entering new {name} chain...\033[0m")  # noqa: T201

    @override
    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain.

        Args:
            outputs: The outputs of the chain.
            **kwargs: Additional keyword arguments.
        """
        print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201

    @override
    def on_agent_action(
        self, action: AgentAction, color: str | None = None, **kwargs: Any
    ) -> Any:
        """Run on agent action.

        Args:
            action: The agent action.
            color: The color to use for the text.
            **kwargs: Additional keyword arguments.
        """
        print_text(action.log, color=color or self.color)

    @override
    def on_tool_end(
        self,
        output: Any,
        color: str | None = None,
        observation_prefix: str | None = None,
        llm_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation.

        Args:
            output: The output to print.
            color: The color to use for the text.
            observation_prefix: The observation prefix.
            llm_prefix: The LLM prefix.
            **kwargs: Additional keyword arguments.
        """
        output = str(output)
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color or self.color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")

    @override
    def on_text(
        self,
        text: str,
        color: str | None = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when the agent ends.

        Args:
            text: The text to print.
            color: The color to use for the text.
            end: The end character to use.
            **kwargs: Additional keyword arguments.
        """
        print_text(text, color=color or self.color, end=end)

    @override
    def on_agent_finish(
        self, finish: AgentFinish, color: str | None = None, **kwargs: Any
    ) -> None:
        """Run on the agent end.

        Args:
            finish: The agent finish.
            color: The color to use for the text.
            **kwargs: Additional keyword arguments.
        """
        print_text(finish.log, color=color or self.color, end="\n")
