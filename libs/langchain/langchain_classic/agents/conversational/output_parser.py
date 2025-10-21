import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.conversational.prompt import FORMAT_INSTRUCTIONS


class ConvoOutputParser(AgentOutputParser):
    """Output parser for the conversational agent."""

    ai_prefix: str = "AI"
    """Prefix to use before AI output."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    def parse(self, text: str) -> AgentAction | AgentFinish:
        """Parse the output from the agent into an AgentAction or AgentFinish object.

        Args:
            text: The text to parse.

        Returns:
            An AgentAction or AgentFinish object.
        """
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()},
                text,
            )
        regex = r"Action: (.*?)[\n]*Action Input: ([\s\S]*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(msg)
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "conversational"
