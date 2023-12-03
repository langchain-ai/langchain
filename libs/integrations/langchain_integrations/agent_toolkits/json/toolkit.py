from __future__ import annotations

from typing import List

from langchain_integrations.agent_toolkits.base import BaseToolkit
from langchain_integrations.tools import BaseTool
from langchain_integrations.tools.json.tool import JsonGetValueTool, JsonListKeysTool, JsonSpec


class JsonToolkit(BaseToolkit):
    """Toolkit for interacting with a JSON spec."""

    spec: JsonSpec

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            JsonListKeysTool(spec=self.spec),
            JsonGetValueTool(spec=self.spec),
        ]
