from langchain.output_parsers.boolean import BooleanOutputParser


class SmartBooleanOutputParser(BooleanOutputParser):
    """Boolean output parser that can handle changes in capitalization, whitespace,
    punctuation, and additional words in the output.

    Works by splitting the output of the language model into words and checking if any of them
    match the true or false values. If multiple words match, the first one is used.

    This output parser is useful when your LLM's give responses like: "Yes, consider 4 reasons
    why...". Since `BooleanOutputParser` only checks for exact matches, it would fail on this
    input. (However if you only want to check for exact matches, `BooleanOutputParser` is what
    you probabbly want.)
    """

    true_val: str = "YES"
    false_val: str = "NO"

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean.

        Args:
            text: output of language model

        Returns:
            boolean

        """
        for word in text.split():
            word = word.strip("\r\n\t \"',.:;!?[]{}|/\\").lower()
            if word == self.true_val.lower():
                return True
            if word == self.false_val.lower():
                return False
        raise ValueError(
            f"BooleanOutputParser expected output value to either contain "
            f"{self.true_val} or {self.false_val}. Got '{text}'."
        )

    @property
    def _type(self) -> str:
        """Snake-case string identifier for output parser type."""
        return "smart_boolean_output_parser"
