from __future__ import annotations

import json
import re
from typing import Union

from pendium.langchain.agents.character_chat.prompt import FORMAT_INSTRUCTIONS

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

FINAL_ANSWER_PREFIX = '''{
"action": "Final Answer",
"action_input": "'''

class ConvoOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        if "```json" in cleaned_output and "```" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
            cleaned_output, _ = cleaned_output.split("```")
        elif "```" in cleaned_output:
            _, cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json") :]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```") :]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()
        try: 
            response = json.loads(cleaned_output)
        except: # Response isn't JSON!
            if cleaned_output.startswith(FINAL_ANSWER_PREFIX): # Probably exhausted tokens, found prefix
                cleaned_output = cleaned_output[len(FINAL_ANSWER_PREFIX) :] 
            return AgentFinish({"output": cleaned_output}, text) # Assume output is final answer
        
        action, action_input = response["action"], response["action_input"]
        if action == "Final Answer":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)