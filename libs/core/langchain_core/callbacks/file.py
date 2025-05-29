"""Callback Handler that writes to a file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TextIO, cast

from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.input import print_text

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


class FileCallbackHandler(BaseCallbackHandler):
    """Callback Handler that writes to a file.

    Parameters:
        filename: The file to write to.
        mode: The mode to open the file in. Defaults to "a".
        color: The color to use for the text.
    """

    def __init__(
        self, filename: str, mode: str = "a", color: Optional[str] = None
    ) -> None:
        """Initialize callback handler.

        Args:
            filename: The filename to write to.
            mode: The mode to open the file in. Defaults to "a".
            color: The color to use for the text. Defaults to None.
        """
        self.file = cast("TextIO", Path(filename).open(mode, encoding="utf-8"))  # noqa: SIM115
        self.color = color

    def __del__(self) -> None:
        """Destructor to cleanup when done."""
        self.file.close()

    @override
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain.

        Args:
            serialized (dict[str, Any]): The serialized chain.
            inputs (dict[str, Any]): The inputs to the chain.
            **kwargs (Any): Additional keyword arguments.
        """
        if "name" in kwargs:
            name = kwargs["name"]
        elif serialized:
            name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        else:
            name = "<unknown>"
        print_text(
            f"\n\n\033[1m> Entering new {name} chain...\033[0m",
            end="\n",
            file=self.file,
        )

    @override
    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain.

        Args:
            outputs (dict[str, Any]): The outputs of the chain.
            **kwargs (Any): Additional keyword arguments.
        """
        print_text("\n\033[1m> Finished chain.\033[0m", end="\n", file=self.file)

    @override
    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action.

        Args:
            action (AgentAction): The agent action.
            color (Optional[str], optional): The color to use for the text.
                Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
        print_text(action.log, color=color or self.color, file=self.file)

    @override
    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation.

        Args:
           output (str): The output to print.
           color (Optional[str], optional): The color to use for the text.
                Defaults to None.
           observation_prefix (Optional[str], optional): The observation prefix.
            Defaults to None.
           llm_prefix (Optional[str], optional): The LLM prefix.
                Defaults to None.
           **kwargs (Any): Additional keyword arguments.
        """
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}", file=self.file)
        print_text(output, color=color or self.color, file=self.file)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}", file=self.file)

    @override
    def on_text(
        self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any
    ) -> None:
        """Run when the agent ends.

        Args:
           text (str): The text to print.
           color (Optional[str], optional): The color to use for the text.
            Defaults to None.
           end (str, optional): The end character. Defaults to "".
           **kwargs (Any): Additional keyword arguments.
        """
        print_text(text, color=color or self.color, end=end, file=self.file)

    @override
    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on the agent end.

        Args:
            finish (AgentFinish): The agent finish.
            color (Optional[str], optional): The color to use for the text.
                Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
        print_text(finish.log, color=color or self.color, end="\n", file=self.file)
