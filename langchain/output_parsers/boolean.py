from langchain.schema import BaseOutputParser


class BooleanOutputParser(BaseOutputParser[bool]):
    true_val: str = "YES"
    false_val: str = "NO"

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean.

        Args:
            text: output of language model

        Returns:
            boolean

        """
        cleaned_text = text.strip()
        if cleaned_text not in (self.true_val, self.false_val):
            raise ValueError(
                f"BooleanOutputParser expected output value to either be "
                f"{self.true_val} or {self.false_val}. Received {cleaned_text}."
            )
        return cleaned_text == self.true_val

    @property
    def _type(self) -> str:
        """Snake-case string identifier for output parser type."""
        return "boolean_output_parser"
