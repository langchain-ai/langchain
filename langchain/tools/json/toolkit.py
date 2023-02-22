from __future__ import annotations

from typing import List

from langchain.tools import BaseTool
from langchain.tools.base import BaseToolkit
from langchain.tools.json.tool import (
    JsonSpec,
    JsonSpecGetValueTool,
    JsonSpecListKeysTool,
)


class JsonSpecToolkit(BaseToolkit):
    """Toolkit for interacting with a JSON spec."""

    spec: JsonSpec

    def get_tools(self) -> List[BaseTool]:
        return [
            JsonSpecListKeysTool(self.spec),
            JsonSpecGetValueTool(self.spec),
        ]
