import re
from typing import List, Optional

from langchain.output_parsers.format_instructions import XML_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser


class XMLOutputParser(BaseOutputParser):
    """Parse an output using xml format."""

    tags: Optional[List[str]] = None
    encoding_matcher: re.Pattern = re.compile(r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL)
    
    def get_format_instructions(self) -> str:
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> str:
        text = text.strip("`").strip("xml")
        encoding_match = self.encoding_matcher.search(text)
        if encoding_match:
            text = encoding_match.group(2)
        if (
            (text.startswith("<") or text.startswith("\n<")) and 
            (text.endswith(">") or text.endswith(">\n"))
        ):
            return text
        else:
            raise ValueError(f"Could not parse output: {text}")

    @property
    def _type(self) -> str:
        return "xml"
