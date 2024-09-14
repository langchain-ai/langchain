"""Toolkit for interacting with Spark SQL."""

from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_community.tools.spark_sql.tool import (
    InfoSparkSQLTool,
    ListSparkSQLTool,
    QueryCheckerTool,
    QuerySparkSQLTool,
)
from langchain_community.utilities.spark_sql import SparkSQL


class SparkSQLToolkit(BaseToolkit):
    """Toolkit for interacting with Spark SQL.

    Parameters:
        db: SparkSQL. The Spark SQL database.
        llm: BaseLanguageModel. The language model.
    """

    db: SparkSQL = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QuerySparkSQLTool(db=self.db),
            InfoSparkSQLTool(db=self.db),
            ListSparkSQLTool(db=self.db),
            QueryCheckerTool(db=self.db, llm=self.llm),
        ]
