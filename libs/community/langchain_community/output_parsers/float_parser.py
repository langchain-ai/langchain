from __future__ import annotations

from typing import Any, Dict
from langchain_core.output_parsers import BaseOutputParser

class FloatOutputParser(BaseOutputParser[float]):
    """Parse the output of an LLM call to a float."""

    @property
    def _type(self) -> str:
        """Return the type of output."""
        return "float"

    def get_format_instructions(self) -> str:
        """Return instructions for how to format the output."""
        return "Your response should be a single float number."

    def parse(self, text: str) -> float:
        """Parse the output of an LLM call to a float.
        
        Args:
            text: The string output of the LLM to parse.
            
        Returns:
            A float value parsed from the text.
            
        Raises:
            ValueError: If the text cannot be parsed into a float.
        """
        try:
            # Clean the text and extract the first float found
            cleaned_text = text.strip().split('\n')[0].strip()
            return float(cleaned_text)
        except ValueError:
            raise ValueError(f"Could not parse float from LLM output: {text}")