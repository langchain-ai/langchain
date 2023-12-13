"""Toolkit for interacting with a SQL database."""
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.sql_database import SQLDatabase
from langchain.tools import BaseTool
from langchain.tools.spark_unitycatalog.tool import (
    InfoUnityCatalogTool,
    ListUnityCatalogTablesTool,
    QueryUCSQLDataBaseTool,
    SqlQueryValidatorTool,
)
from langchain.tools.sql_database.tool import QuerySQLCheckerTool
from pydantic import Field


class UCSQLDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    db_token: str
    db_host: str
    db_catalog: str
    db_schema: str
    db_warehouse_id: str
    allow_extra_fields = True

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
            QueryUCSQLDataBaseTool(
                db=self.db, description=query_sql_database_tool_description
            ),
            InfoUnityCatalogTool(
                db=self.db,
                description=info_sql_database_tool_description,
                db_token=self.db_token,
                db_host=self.db_host,
                db_catalog=self.db_catalog,
                db_schema=self.db_schema,
                db_warehouse_id=self.db_warehouse_id,
            ),
            ListUnityCatalogTablesTool(
                db=self.db,
                db_token=self.db_token,
                db_host=self.db_host,
                db_catalog=self.db_catalog,
                db_schema=self.db_schema,
                db_warehouse_id=self.db_warehouse_id,
            ),
            QuerySQLCheckerTool(db=self.db, llm=self.llm),
            SqlQueryValidatorTool(llm = self.llm),
        ]
