import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from langchain.output_parsers.format_instructions import XML_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser


class XMLOutputParser(BaseOutputParser):
    """Parse an output using xml format."""

    tags: Optional[List[str]] = None
    encoding_matcher: re.Pattern = re.compile(
        r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL
    )

    def get_format_instructions(self) -> str:
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> Dict[str, List[Any]]:
        text = text.strip("`").strip("xml")
        encoding_match = self.encoding_matcher.search(text)
        if encoding_match:
            text = encoding_match.group(2)
        if (text.startswith("<") or text.startswith("\n<")) and (
            text.endswith(">") or text.endswith(">\n")
        ):
            root = ET.fromstring(text)
            return self._root_to_dict(root)
        else:
            raise ValueError(f"Could not parse output: {text}")

    def _root_to_dict(self, root: ET.Element) -> Dict[str, List[Any]]:
        """Converts xml tree to python dictionary."""
        result: Dict[str, List[Any]] = {root.tag: []}
        for child in root:
            if len(child) == 0:
                result[root.tag].append({child.tag: child.text})
            else:
                result[root.tag].append(self._root_to_dict(child))
        return result

    @property
    def _type(self) -> str:
        return "xml"
