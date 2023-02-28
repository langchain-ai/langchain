"""Toolkit for interacting with a vector store."""
from typing import List, Union

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.tools.vectorstore.tool import VectorStoreQATool, VectorStoreQAWithSourcesTool


class VectorStoreToolkit(BaseToolkit):
    """Toolkit for interacting with a vector store."""

    tools: List[Union[VectorStoreQATool, VectorStoreQAWithSourcesTool]] = []

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
