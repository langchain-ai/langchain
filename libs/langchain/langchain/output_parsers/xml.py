import re
from typing import List, Optional

from langchain.output_parsers.format_instructions import XML_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser


class XMLOutputParser(BaseOutputParser):
    """Parse an output using xml format."""

    tags: Optional[List[str]] = None

    def get_format_instructions(self) -> str:
        if self.tags:
            return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)
        return """Write a following string formatted as an XML file.
        Always mention the encoding of the file in the first line.
        Remember to open and close all tags.
        """

    def parse(self, text: str) -> str:
        encoding_match = re.search(
            r"<([^>]*encoding[^>]*)>\n(.*)", text, re.MULTILINE | re.DOTALL
        )
        if encoding_match:
            text = encoding_match.group(2)
        if text.startswith("<") and text.endswith(">"):
            return text
        else:
            raise ValueError(f"Could not parse output: {text}")

    @property
    def _type(self) -> str:
        return "xml"
