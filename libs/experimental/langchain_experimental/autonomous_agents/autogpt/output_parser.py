import json
import re
from abc import abstractmethod
from typing import Dict, NamedTuple

from langchain.schema import BaseOutputParser


class AutoGPTAction(NamedTuple):
    """Action returned by AutoGPTOutputParser."""

    name: str
    args: Dict


class BaseAutoGPTOutputParser(BaseOutputParser):
    """Base Output parser for AutoGPT."""

    @abstractmethod
    def parse(self, text: str) -> AutoGPTAction:
        """Return AutoGPTAction"""


def preprocess_json_input(input_str: str) -> str:
    """Preprocesses a string to be parsed as json.

    Replace single backslashes with double backslashes,
    while leaving already escaped ones intact.

    Args:
        input_str: String to be preprocessed

    Returns:
        Preprocessed string
    """
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
    )
    return corrected_str


class AutoGPTOutputParser(BaseAutoGPTOutputParser):
    """Output parser for AutoGPT."""

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
