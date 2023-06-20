"""Toolkit for interacting with a SQL database."""
from typing import List

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.sql_database import SQLDatabase
from langchain.tools import BaseTool
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)


class SQLDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            "'xxxx' in 'field list', using schema_sql_db to query the correct table "
            "fields."
        )
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling list_tables_sql_db "
            "first! Example Input: 'table1, table2, table3'"
        )
        return [
            QuerySQLDataBaseTool(
                db=self.db, description=query_sql_database_tool_description
            ),
            InfoSQLDatabaseTool(
                db=self.db, description=info_sql_database_tool_description
            ),
            ListSQLDatabaseTool(db=self.db),
            QuerySQLCheckerTool(db=self.db, llm=self.llm),
        ]
