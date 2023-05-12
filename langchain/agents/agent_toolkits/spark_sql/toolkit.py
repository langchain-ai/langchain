"""Toolkit for interacting with a SQL database."""
from typing import List

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.spark_sql import SparkSQL
from langchain.tools import BaseTool
from langchain.tools.spark_sql.tool import (
    InfoSparkSQLTool,
    ListSparkSQLTool,
    QueryCheckerTool,
    QuerySparkSQLTool,
)


class SparkSQLToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SparkSQL = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QuerySparkSQLTool(db=self.db),
            InfoSparkSQLTool(db=self.db),
            ListSparkSQLTool(db=self.db),
            QueryCheckerTool(db=self.db, llm=self.llm),
        ]
