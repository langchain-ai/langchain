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
        cleaned_text = text.strip()
        if cleaned_text.upper() not in (self.true_val.upper(), self.false_val.upper()):
            raise ValueError(
                f"BooleanOutputParser expected output value to either be "
                f"{self.true_val} or {self.false_val}. Received {cleaned_text}."
            )
        return cleaned_text.upper() == self.true_val.upper()

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "boolean_output_parser"
