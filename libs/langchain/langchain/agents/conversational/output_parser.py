import re
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS


class ConvoOutputParser(AgentOutputParser):
    """Output parser for the conversational agent."""

    ai_prefix: str = "AI"
    """Prefix to use before AI output."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        regex = r"Action: (.*?)[\n]*Action Input: ([\s\S]*)"
        match = re.search(regex, text)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1)
        action_input = match.group(2)
        # Generate all partials of "\nObservation: " down to "\nO"
        observation_base = "\nObservation: "
        parts = action_input.rsplit(observation_base, 1)
        if len(parts) == 2:
            action_input = parts[0]
        for i in range(len(observation_base) - 1, 1, -1):
            partial = observation_base[:i]
            if action_input.endswith(partial):
                action_input = action_input.rsplit(partial, 1)[0]
                break
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "conversational"
