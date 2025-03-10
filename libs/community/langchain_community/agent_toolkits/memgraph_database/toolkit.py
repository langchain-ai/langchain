"""Memgraph graph database Toolkit."""

from typing import List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_community.tools.memgraph.tool import (
    GetSchemaMemgraphDatabaseTool,
)

from langchain_community.utilities.memgraph import Memgraph


class MemgraphDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with Memgraph database.

    Parameters:
        db: Memgraph database instance to interact with .
    """

    db: Memgraph = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            GetSchemaMemgraphDatabaseTool(db=self.db)
        ]
