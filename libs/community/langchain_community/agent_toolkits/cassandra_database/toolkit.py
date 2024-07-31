"""Apache Cassandra Toolkit."""

from typing import List

from pydantic import ConfigDict, Field

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.cassandra_database.tool import (
    GetSchemaCassandraDatabaseTool,
    GetTableDataCassandraDatabaseTool,
    QueryCassandraDatabaseTool,
)
from langchain_community.utilities.cassandra_database import CassandraDatabase


class CassandraDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with an Apache Cassandra database.

    Parameters:
        db: CassandraDatabase. The Cassandra database to interact
            with.
    """

    db: CassandraDatabase = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            GetSchemaCassandraDatabaseTool(db=self.db),
            QueryCassandraDatabaseTool(db=self.db),
            GetTableDataCassandraDatabaseTool(db=self.db),
        ]
