import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class ReActOutputParser(AgentOutputParser):
    """Output parser for the ReAct agent."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            raise OutputParserException(f"Could not parse LLM Output: {text}")
        action_block = text.strip().split("\n")[-1]

        action_str = action_block[len(action_prefix) :]
        # Parse out the action and the directive.
        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            raise OutputParserException(
                f"Could not parse action directive: {action_str}"
            )
        action, action_input = re_matches.group(1), re_matches.group(2)
        if action == "Finish":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)

    @property
    def _type(self) -> str:
        return "react"
