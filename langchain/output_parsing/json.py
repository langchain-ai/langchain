"""Parse json output."""
import json
from typing import Dict, List, Union

from langchain.output_parsing.base import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    """Parse json output."""

    def parse(self, text: str) -> Union[str, List[str], Dict[str, str]]:
        """Parse json string."""
        return json.loads(text)
