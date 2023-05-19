from __future__ import annotations

import json
import re
from typing import Union

from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException

FINAL_ANSWER_PREFIX = '{\n"action": "Final Answer",\n"action_input": "'

class ConvoOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            cleaned_output = text.strip()
            cleaned_output = re.sub(r"\n+", "\n", cleaned_output)

            # clean JSON
            if "```json" in cleaned_output and cleaned_output.count("```") == 2:
                _, cleaned_output = cleaned_output.split("```json")
                cleaned_output, _ = cleaned_output.split("```")
            # markdown but not labeled as JSON
            if cleaned_output.count("```") == 2: 
                _, cleaned_output, _ = cleaned_output.split("```")
            
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[len("```json"):]
            if cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[len("```"):]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[: -len("```")]
            cleaned_output = cleaned_output.strip()
            try:
                response = json.loads(cleaned_output)
            except json.JSONDecodeError as e:
                if "action" not in cleaned_output or FINAL_ANSWER_PREFIX in cleaned_output:
                    if cleaned_output.startswith(FINAL_ANSWER_PREFIX): # Found prefix, probably exhausted tokens
                        cleaned_output = cleaned_output[len(FINAL_ANSWER_PREFIX):]
                    return AgentFinish({"output": cleaned_output}, text)
                raise OutputParserException(f"Got invalid JSON object: {text}") from e

            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "conversational_chat"
