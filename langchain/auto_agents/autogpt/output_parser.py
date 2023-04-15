import json
from typing import Dict, NamedTuple, Union

from langchain.schema import AgentAction, AgentFinish, BaseOutputParser


class AutoGPTAction(NamedTuple):
    name: str
    args: Dict


class BaseAutoGPTOutputParser(BaseOutputParser):
    def parse(self, text: str) -> AutoGPTAction:
        """Return AutoGPTAction"""


class AutoGPTOutputParser(BaseAutoGPTOutputParser):
    def parse(self, text: str) -> AutoGPTAction:
        parsed = json.loads(text)
        return AutoGPTAction(
            name=parsed["command"]["name"],
            args=parsed["command"]["args"],
        )
