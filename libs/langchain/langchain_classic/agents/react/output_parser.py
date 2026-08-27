import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser


class ReActOutputParser(AgentOutputParser):
    """Output parser for the ReAct agent."""

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            msg = f"Could not parse LLM Output: {text}"
            raise OutputParserException(msg)
        action_block = text.strip().split("\n")[-1]

        action_str = action_block[len(action_prefix) :]
        # Parse out the action and the directive.
        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            msg = f"Could not parse action directive: {action_str}"
            raise OutputParserException(msg)
        action, action_input = re_matches.group(1), re_matches.group(2)
        if action == "Finish":
            return AgentFinish({"output": action_input}, text)
        return AgentAction(action, action_input, text)

    @property
    def _type(self) -> str:
        return "react"
