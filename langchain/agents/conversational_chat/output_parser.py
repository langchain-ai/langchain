from __future__ import annotations

import json
import re
from typing import Union

from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish


class ConvoOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()

        action_pattern = r'"action":\s*"([^"]*)"'
        action_input_pattern = r'"action_input":\s*"([^"]*)"'

        action_match = re.search(action_pattern, cleaned_output)
        action_input_match = re.search(action_input_pattern, cleaned_output)

        try:
            if action_match is None or action_input_match is None:
                raise ValueError(
                    "Failed to parse values from the LLM output: ", cleaned_output
                )

            action = action_match.group(1)
            action_input = action_input_match.group(1)

            parsed = {"action": action, "action_input": action_input}
            parsed_output = json.dumps(parsed)

        except AttributeError:
            print("Failed to parse LLM output: ", cleaned_output)

        response = json.loads(parsed_output)
        action, action_input = response["action"], response["action_input"]
        if action == "Final Answer":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)
