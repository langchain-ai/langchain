"""Callback Handler that writes to a file."""
from typing import Any, Dict, Optional, TextIO, cast

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.utils.input import print_text

from langchain.callbacks.base import BaseCallbackHandler


class FileCallbackHandler(BaseCallbackHandler):
    """Callback Handler that writes to a file."""

    def __init__(
        self, filename: str, mode: str = "a", color: Optional[str] = None
    ) -> None:
        """Initialize callback handler."""
        self.file = cast(TextIO, open(filename, mode, encoding="utf-8"))
        self.color = color

    def __del__(self) -> None:
        """Destructor to cleanup when done."""
        self.file.close()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print_text(
            f"\n\n\033[1m> Entering new {class_name} chain...\033[0m",
            end="\n",
            file=self.file,
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print_text("\n\033[1m> Finished chain.\033[0m", end="\n", file=self.file)

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print_text(action.log, color=color or self.color, file=self.file)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}", file=self.file)
        print_text(output, color=color or self.color, file=self.file)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}", file=self.file)

    def on_text(
        self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any
    ) -> None:
        """Run when agent ends."""
        print_text(text, color=color or self.color, end=end, file=self.file)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print_text(finish.log, color=color or self.color, end="\n", file=self.file)
