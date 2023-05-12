from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class ReflexionOutputParser(AgentOutputParser):
    def parse(self, text: str) -> str:
        # The Reflexion prompt asks the LLM to complete after "New plan: ",
        # so the entire result is the reflexion
        return text
