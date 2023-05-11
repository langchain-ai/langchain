from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class ReflexionOutputParser(AgentOutputParser):
    def parse(self, text: str) -> str:
        result_prefix = "New plan: "

        # We expect last line of LLM output to start with "New plan: "
        if not text.strip().split("\n")[-1].startswith(result_prefix):
            raise OutputParserException(f"Could not parse LLM Output: {text}")
        
        result_line = text.strip().split("\n")[-1]

        # Everything after that is the reflexion output
        return result_line[len(result_prefix) :]
