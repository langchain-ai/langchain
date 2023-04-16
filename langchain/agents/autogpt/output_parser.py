import json
from abc import abstractmethod
from typing import Dict, NamedTuple

from langchain.schema import BaseOutputParser


class AutoGPTAction(NamedTuple):
    name: str
    args: Dict


class BaseAutoGPTOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> AutoGPTAction:
        """Return AutoGPTAction"""


class AutoGPTOutputParser(BaseAutoGPTOutputParser):
    def parse(self, text: str) -> AutoGPTAction:
        parsed = json.loads(text)
        return AutoGPTAction(
            name=parsed["command"]["name"],
            args=parsed["command"]["args"],
        )
