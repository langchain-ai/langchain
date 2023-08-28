import re

from langchain.schema import BaseOutputParser


class XMLOutputParser(BaseOutputParser):
    """Parse an output using xml format."""
    
    def get_format_instructions(self) -> str:
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