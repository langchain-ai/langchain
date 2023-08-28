from typing import List, Optional

import re

from langchain.output_parsers.format_instructions import XML_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser


class XMLOutputParser(BaseOutputParser):
    """Parse an output using xml format."""
    
    tags: Optional[List[str]] = None
    
    def get_format_instructions(self) -> str:
        if self.tags:
            return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)
        return f"""Write a following string formatted as an XML file.
        Always mention the encoding of the file in the first line.
        Remember to open and close all tags.
        """
    
    def parse(self, text: str) -> str:
        match = re.search(r'\n(.*)', text, re.MULTILINE | re.DOTALL)
        if match:
            return match.group()
        else:
            raise ValueError(f"Could not parse output: {text}")
    
    @property
    def _type(self) -> str:
        return "xml"