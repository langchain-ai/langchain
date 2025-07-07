from __future__ import annotations

from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS


# Define a class that parses output for conversational agents
class ConvoOutputParser(AgentOutputParser):
    """Output parser for the conversational agent."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into an AgentAction or AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        try:
            # Attempt to parse the text into a structured format (assumed to be JSON
            # stored as markdown)
            response = parse_json_markdown(text)

            # If the response contains an 'action' and 'action_input'
            if "action" in response and "action_input" in response:
                action, action_input = response["action"], response["action_input"]

                # If the action indicates a final answer, return an AgentFinish
                if action == "Final Answer":
                    return AgentFinish({"output": action_input}, text)
                # Otherwise, return an AgentAction with the specified action and
                # input
                return AgentAction(action, action_input, text)
            # If the necessary keys aren't present in the response, raise an
            # exception
            msg = f"Missing 'action' or 'action_input' in LLM output: {text}"
            raise OutputParserException(msg)
        except Exception as e:
            # If any other exception is raised during parsing, also raise an
            # OutputParserException
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "conversational_chat"
