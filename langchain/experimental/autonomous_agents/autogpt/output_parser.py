import json
import re
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


def preprocess_json_input(input_str: str) -> str:
    # Replace single backslashes with double backslashes,
    # while leaving already escaped ones intact
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
    )
    return corrected_str


class AutoGPTOutputParser(BaseAutoGPTOutputParser):
    def parse(self, text: str) -> AutoGPTAction:
        try:
            parsed = json.loads(text, strict=False)
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(text)
            try:
                parsed = json.loads(preprocessed_text, strict=False)
            except Exception:
                return AutoGPTAction(
                    name="ERROR",
                    args={"error": f"Could not parse invalid json: {text}"},
                )
        try:
            return AutoGPTAction(
                name=parsed["command"]["name"],
                args=parsed["command"]["args"],
            )
        except (KeyError, TypeError):
            # If the command is null or incomplete, return an erroneous tool
            return AutoGPTAction(
                name="ERROR", args={"error": f"Incomplete command args: {parsed}"}
            )
