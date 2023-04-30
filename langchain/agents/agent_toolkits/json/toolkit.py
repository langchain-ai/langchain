"""Toolkit for interacting with a JSON spec."""
from __future__ import annotations

from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.tools.json.tool import JsonGetValueTool, JsonListKeysTool, JsonSpec


class JsonToolkit(BaseToolkit):
    """Toolkit for interacting with a JSON spec."""

    spec: JsonSpec

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            JsonListKeysTool(spec=self.spec),
            JsonGetValueTool(spec=self.spec),
        ]
