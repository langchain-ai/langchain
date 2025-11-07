import re

from langchain_core.output_parsers import BaseOutputParser


class BooleanOutputParser(BaseOutputParser[bool]):
    """Parse the output of an LLM call to a boolean."""

    true_val: str = "YES"
    """The string value that should be parsed as True."""
    false_val: str = "NO"
    """The string value that should be parsed as False."""

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean.

        Args:
            text: output of a language model

        Returns:
            boolean
        """
        regexp = rf"\b({self.true_val}|{self.false_val})\b"

        truthy = {
            val.upper()
            for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
        }
        if self.true_val.upper() in truthy:
            if self.false_val.upper() in truthy:
                msg = (
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return True
        if self.false_val.upper() in truthy:
            if self.true_val.upper() in truthy:
                msg = (
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return False
        msg = (
            f"BooleanOutputParser expected output value to include either "
            f"{self.true_val} or {self.false_val}. Received {text}."
        )
        raise ValueError(msg)

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "boolean_output_parser"
