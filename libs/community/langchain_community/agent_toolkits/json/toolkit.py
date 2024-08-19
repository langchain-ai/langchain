from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.json.tool import (
    JsonGetValueTool,
    JsonListKeysTool,
    JsonSpec,
)


class JsonToolkit(BaseToolkit):
    """Toolkit for interacting with a JSON spec.

    Parameters:
        spec: The JSON spec.
    """

    spec: JsonSpec

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            JsonListKeysTool(spec=self.spec),
            JsonGetValueTool(spec=self.spec),
        ]
