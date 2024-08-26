"""Toolkit for interacting with an SQL database."""

from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
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
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Входными данными для этого инструмента является список таблиц, "
            "разделённый запятыми, на выходе — "
            "схема и образцы строк для этих таблиц. "
            "Убедись, что таблицы действительно существуют, сначала вызвав "
            f"{list_sql_database_tool.name}! "
            'Параметры передавай в виде JSON: ["table1", "table2", "table3"]. '
            "Если данных в таблице недостаточно запроси другие таблицы"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Входными данными для этого инструмента является подробный "
            "и корректный SQL-запрос, на выходе — "
            "результат из базы данных. Если запрос некорректен, "
            "будет возвращено сообщение об ошибке. "
            "Если вернулась ошибка, перепиши запрос, проверь его "
            "и попробуй снова. Если возникнет ошибка "
            f"с сообщением 'Unknown column 'xxxx' in 'field list'', "
            f"используй {info_sql_database_tool.name}, "
            "чтобы запросить правильные поля таблицы."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Используй этот инструмент, чтобы проверить правильность "
            "своего запроса перед его выполнением. "
            "Всегда используй этот инструмент перед выполнением запроса с помощью "
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
