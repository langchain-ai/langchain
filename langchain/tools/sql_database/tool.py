# flake8: noqa
"""Tools for interacting with a SQL database."""
from pydantic import BaseModel, Extra, Field, validator, root_validator
from typing import Any, Dict

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.llms.base import BaseLLM
from langchain.tools.base import BaseTool
from langchain.tools.sql_database.prompt import QUERY_CHECKER


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    # Override BaseTool.Config to appease mypy
    # See https://github.com/pydantic/pydantic/issues/4173
    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""

    name = "query_sql_db"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "schema_sql_db"
    description = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling list_tables_sql_db first!

    Example Input: "table1, table2, table3"
    """

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(table_names.split(", "))

    async def _arun(self, table_name: str) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting tables names."""

    name = "list_tables_sql_db"
    description = "Input is an empty string, output is a comma separated list of tables in the database."

    def _run(self, tool_input: str = "") -> str:
        """Get the schema for a specific table."""
        return ", ".join(self.db.get_usable_table_names())

    async def _arun(self, tool_input: str = "") -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class QueryCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: BaseLLM
    llm_chain: LLMChain = Field(init=False)
    name = "query_checker_sql_db"
    description = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    """

    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "llm_chain" not in values:
            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),
                prompt=PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query", "dialect"]
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["query", "dialect"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )

        return values

    def _run(self, query: str) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(query=query, dialect=self.db.dialect)

    async def _arun(self, query: str) -> str:
        return await self.llm_chain.apredict(query=query, dialect=self.db.dialect)
