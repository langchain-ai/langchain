"""Toolkit for interacting with an SQL database."""

from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase


class SQLDatabaseToolkit(BaseToolkit):
    """SQLDatabaseToolkit for interacting with SQL databases.

    Setup:
        Install ``langchain-community``.

        .. code-block:: bash

            pip install -U langchain-community

    Key init args:
        db: SQLDatabase
            The SQL database.
        llm: BaseLanguageModel
            The language model (for use with QuerySQLCheckerTool)

    Instantiate:
        .. code-block:: python

            from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            from langchain_community.utilities.sql_database import SQLDatabase
            from langchain_openai import ChatOpenAI

            db = SQLDatabase.from_uri("sqlite:///Chinook.db")
            llm = ChatOpenAI(temperature=0)

            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    Tools:
        .. code-block:: python

            toolkit.get_tools()

    Use within an agent:
        .. code-block:: python

            from langchain import hub
            from langgraph.prebuilt import create_react_agent

            # Pull prompt (or define your own)
            prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
            system_message = prompt_template.format(dialect="SQLite", top_k=5)

            # Create agent
            agent_executor = create_react_agent(
                llm, toolkit.get_tools(), state_modifier=system_message
            )

            # Query agent
            example_query = "Which country's customers spent the most?"

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()
    """  # noqa: E501

    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return self.db.dialect

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        )
        return [
            query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            query_sql_checker_tool,
        ]

    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()
