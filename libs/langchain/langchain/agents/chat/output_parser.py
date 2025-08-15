import json
import re
from re import Pattern
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.agent import AgentOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"


class ChatOutputParser(AgentOutputParser):
    """Output parser for the chat agent."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""

    pattern: Pattern = re.compile(r"^.*?`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL)
    """Regex pattern to parse the output."""

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the output from the agent into
        an AgentAction or AgentFinish object.

        Args:
            text: The text to parse.

        Returns:
            An AgentAction or AgentFinish object.

        Raises:
            OutputParserException: If the output could not be parsed.
            ValueError: If the action could not be found.
        """

        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            found = self.pattern.search(text)
            if not found:
                # Fast fail to parse Final Answer.
                msg = "action not found"
                raise ValueError(msg)
            action = found.group(1)
            response = json.loads(action.strip())
            includes_action = "action" in response
            if includes_answer and includes_action:
                msg = (
                    "Parsing LLM output produced a final answer "
                    f"and a parse-able action: {text}"
                )
                raise OutputParserException(msg)
            return AgentAction(
                response["action"],
                response.get("action_input", {}),
                text,
            )

        except Exception as exc:
            if not includes_answer:
                msg = f"Could not parse LLM output: {text}"
                raise OutputParserException(msg) from exc
            output = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": output}, text)

    @property
    def _type(self) -> str:
        return "chat"
