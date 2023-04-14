from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import json


class AutoGPTOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        parsed = json.loads(text)
        return AgentAction(tool=parsed["command"]["name"], tool_input=parsed["command"]["input"], log=text)
