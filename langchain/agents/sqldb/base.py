import string
from typing import Any, List, Optional, Sequence

import sqlalchemy
from pydantic import BaseModel, BaseSettings, Field, root_validator, validator
from sqlalchemy import create_engine, text

from langchain import PromptTemplate
from langchain.agents import ZeroShotAgent
from langchain.agents.mrkl.base import create_zero_shot_prompt
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.sqldb.prompt import PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM
from langchain.sql_database import SQLDatabase
from langchain.tools.base import BaseTool, BaseToolkit


class BaseSQLDBTool(BaseModel):
    """Implement a tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class QuerySqlDbTool(BaseSQLDBTool, BaseTool):
    """Tool for querying a SQL database."""

    name = "query_sql_db"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")


class SchemaSqlDbTool(BaseSQLDBTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "schema_sql_db"
    description = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be SURE that the tables actually exist by calling list_tables_sql_db first!
    
    Example Input: "table1, table2, table3"
    """

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info(table_names.split(", "))

    async def _arun(self, table_name: str) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ListTablesSqlDbTool(BaseSQLDBTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "list_tables_sql_db"
    description = "Input is an empty string, output is a comma separated list of tables in the database."

    def _run(self, tool_input: str = "") -> str:
        """Get the schema for a specific table."""
        return ", ".join(self.db.get_table_names())

    async def _arun(self, tool_input: str = "") -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class QueryCheckerTool(BaseSQLDBTool, BaseTool):
    """Use an LLM to check if a query is correct."""

    template = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Using the correct number of arguments for operators
- Casting to the correct data type
- Using the proper columns for joins

Rewrite the query above if there are any mistakes. If it looks good as it is, just reproduce the original query."""
    llm_chain: LLMChain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=PromptTemplate(template=template, input_variables=["query", "dialect"]),
    )
    name = "query_checker_sql_db"
    description = "Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with query_sql_db!"

    def _run(self, query: str) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(query=query, dialect=self.db.dialect)

    async def _arun(self, query: str) -> str:
        return await self.llm_chain.apredict(query=query, dialect=self.db.dialect)


class SQLDBToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QuerySqlDbTool(db=self.db),
            SchemaSqlDbTool(db=self.db),
            ListTablesSqlDbTool(db=self.db),
            QueryCheckerTool(db=self.db),
        ]


class SqlDatabaseAgent(ZeroShotAgent):
    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        dialect: str = "sqlite",
        top_k: int = 10,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            dialect: The dialect of the SQL database.
            top_k: The number of results to return.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            format_instructions: Instructions to put before the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        prefix = string.Formatter().format(prefix, dialect=dialect, top_k=top_k)
        return create_zero_shot_prompt(
            format_instructions, input_variables, prefix, suffix, tools
        )

    @classmethod
    def from_llm_and_toolkit(
        cls,
        llm: BaseLLM,
        toolkit: SQLDBToolkit,
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        top_k: int = 10,
        **kwargs: Any,
    ):
        """Create an agent from an LLM, database, and tools."""
        tools = toolkit.get_tools()
        prompt = cls.create_prompt(
            tools,
            dialect=db.dialect,
            top_k=top_k,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)


if __name__ == "__main__":
    from langchain.agents import AgentExecutor, ZeroShotAgent

    db = SQLDatabase.from_uri(
        "postgresql+pg8000://postgres:postgres@localhost:5432/postgres"
    )
    toolkit = SQLDBToolkit(db=db)

    agent = SqlDatabaseAgent.from_llm_and_toolkit(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=toolkit.get_tools(), verbose=True
    )
    agent_executor.run("How many tools runs and llm runs did user 1 make today?")
