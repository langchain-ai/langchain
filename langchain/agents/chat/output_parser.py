import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException

FINAL_ANSWER_ACTION = "Final Answer:"


class ChatOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        try:
            action_pattern = r'"action":\s*"([^"]*)"'
            action_input_pattern = r'"action_input":\s*"([^"]*)"'
            action_match = re.search(action_pattern, text)
            action_input_match = re.search(action_input_pattern, text)

            if action_match and action_input_match:
                action = action_match.group(1)
                action_input = action_input_match.group(1)
                return AgentAction(action, action_input, text)
            else:
                raise OutputParserException(f"Could not parse LLM output: {text}")

        except Exception:
            raise OutputParserException(f"Could not parse LLM output: {text}")

    @property
    def _type(self) -> str:
        return "chat"
