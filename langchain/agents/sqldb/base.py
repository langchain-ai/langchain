import os
os.environ["LANGCHAIN_HANDLER"] = "langchain"
from langchain.tools.base import BaseTool, BaseToolkit
import sqlalchemy
from sqlalchemy import create_engine
from pydantic import BaseModel, BaseSettings, validator, root_validator
from sqlalchemy import text
from typing import Any, List
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate


class SQLDBSettings(BaseSettings):
    """Settings for SQLDBTool."""
    SQLALCHEMY_DATABASE_URI: str = "postgresql+pg8000://postgres:postgres@localhost:5432/postgres"


class BaseSQLDBTool(BaseModel):
    """Implement a tool for interacting with a SQL database."""

    dialect: str = "postgresql"
    settings: SQLDBSettings = SQLDBSettings()
    engine: Any
    metadata: Any

    class Config:
        arbitrary_types_allowed = True

    @validator("dialect")
    def validate_dialect_sqlite_postgresql_mysql(cls, v: str) -> str:
        if v not in ["sqlite", "postgresql", "mysql"]:
            raise ValueError("dialect must be one of sqlite, postgresql, mysql")
        return v

    @root_validator
    def validate_engine_metadata(cls, values: dict) -> dict:
        """Create the engine and metadata."""
        values["engine"] = create_engine(values["settings"].SQLALCHEMY_DATABASE_URI)
        values["metadata"] = sqlalchemy.MetaData()
        values["metadata"].reflect(values["engine"])
        return values


class QuerySqlDbTool(BaseSQLDBTool, BaseTool):
    """Tool for querying a SQL database."""

    name = "query_sql_db"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned. If an error is returned, rewrite the query and try again!
    """
    limit: int = 50

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        result = ""
        try:
            with self.engine.connect() as conn:
                cursor = conn.execute(text(query))
                for r in cursor:
                    result += str(r) + "\n"
                return result
        except Exception as e:
            """Format the error message."""
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")


class SchemaSqlDbTool(BaseSQLDBTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "schema_sql_db"
    description = """
    Input to this tool is a table name, output is the schema for that table. Be SURE that the table actually exists by calling list_tables_sql_db first!
    """

    def _run(self, table_name: str) -> str:
        """Get the schema for a specific table."""
        if table_name not in self.metadata.tables:
            return f"Error: {table_name} does not exist in the database. Try again."
        return self.metadata.tables[table_name].__repr__()

    async def _arun(self, table_name: str) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ListTablesSqlDbTool(BaseSQLDBTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "list_tables_sql_db"
    description = "Input is an empty string, output is a list of tables in the database."

    def _run(self, tool_input: str ="") -> str:
        """Get the schema for a specific table."""
        return "\n".join(self.metadata.tables.keys())

    async def _arun(self, tool_input: str = "") -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class QueryCheckerTool(BaseSQLDBTool, BaseTool):
    """Use an LLM to check if a query is correct."""

    template = """{query}
    Double check the {dialect} query above for common mistakes, including:
     - Using NOT IN with NULL values
     - Using UNION when UNION ALL should have been used
     - Using BETWEEN for exclusive ranges
     - Data type mismatch in predicates
     - Properly quoting identifiers
     - Using the correct number of arguments for functions
     - Using the correct number of arguments for operators
     - Using the correct number of arguments for CASE
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
        return self.llm_chain.predict(query=query, dialect=self.dialect)

    async def _arun(self, query: str) -> str:
        return await self.llm_chain.apredict(query=query, dialect=self.dialect)


class SQLDBToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [QuerySqlDbTool(), SchemaSqlDbTool(), ListTablesSqlDbTool(), QueryCheckerTool()]


if __name__ == "__main__":
    from langchain.agents import ZeroShotAgent, AgentExecutor

    prefix = """
You are an agent designed to interact with a SQL database.
Your goal is to return a final answer by interacting with the SQL database.
You have access to a toolkit that contains tools for interacting with the database.
You can use the tools to query the database, get the schema for a table, and check if a query is correct, and get the tables in the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
    """
    suffix = """
Begin!"

Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}"""

    tools = SQLDBToolkit().get_tools()
    os.environ["SQLALCHEMY_DATABASE_URI"] = "postgresql+pg8000://postgres:postgres@localhost:5432/postgres"

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )
    tool_names = [tool.name for tool in tools]
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    agent_executor.run("How many tools runs did user 1 make today?")

