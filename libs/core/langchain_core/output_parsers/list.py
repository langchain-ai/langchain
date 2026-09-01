class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list."""

    pattern: str = r"^\s*(\d+)\.\s([^\n]+)$"
    """The pattern to match a numbered list item."""

    @override
    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """
        return re.findall(self.pattern, text, re.MULTILINE)

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "numbered-list"
