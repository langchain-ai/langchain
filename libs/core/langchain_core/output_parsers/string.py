"""String output parser."""

from typing_extensions import override

from langchain_core.output_parsers.transform import BaseTransformOutputParser


class StrOutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses `LLMResult` into the top likely string."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`StrOutputParser` is serializable.

        Returns:
            `True`
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "output_parser"]`
        """
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "default"

    @override
    def parse(self, text: str) -> str:
        """Returns the input text with no changes."""
        return text
