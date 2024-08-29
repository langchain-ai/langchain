# flake8: noqa
"""Tools for interacting with a SQL database."""

import json
from typing import Any, Dict, Optional, Sequence, Type, Union

from sqlalchemy.engine import Result

from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_core.utils.json import parse_partial_json


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    class Config(BaseTool.Config):
        pass


class _QuerySQLDataBaseToolInput(BaseModel):
    query: str = Field(..., description="Подробный и корректный SQL-запрос.")


class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""

    name: str = "sql_db_query"
    description: str = """
    Выполни SQL-запрос к базе данных и получи результат.
    Если запрос некорректен, будет возвращено сообщение об ошибке.
    Если вернулась ошибка, перепиши запрос, проверь его и попробуй снова.
    """
    args_schema: Type[BaseModel] = _QuerySQLDataBaseToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)


class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "Список имён таблиц, для которых нужно вернуть схему, разделённый запятыми. "
            "Пример ввода: 'table1, table2, table3'"
        ),
    )


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Получает схему и образцы строк для указанных SQL-таблиц."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        try:
            return self.db.get_table_info_no_throw(parse_partial_json(table_names))
        except json.JSONDecodeError:
            return "Не валидный JSON!"


class _ListSQLDataBaseToolInput(BaseModel):
    tool_input: str = Field("", description="Пустая строка")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting tables names."""

    name: str = "sql_db_list_tables"
    description: str = (
        "Входные данные — пустая строка, "
        "выходные данные — список таблиц в базе данных, разделённый запятыми."
    )
    args_schema: Type[BaseModel] = _ListSQLDataBaseToolInput

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a comma-separated list of table names."""
        return json.dumps(self.db.get_usable_table_names(), ensure_ascii=False)


class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="Подробный SQL-запрос для проверки.")


class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "sql_db_query_checker"
    description: str = """
    Используй этот инструмент, чтобы дважды проверить правильность своего запроса перед его выполнением.
    Всегда используй этот инструмент перед выполнением запроса с помощью sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "llm_chain" not in values:
            from langchain.chains.llm import LLMChain

            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),  # type: ignore[arg-type]
                prompt=PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["dialect", "query"]
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["dialect", "query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            query=query,
            dialect=self.db.dialect,
            callbacks=run_manager.get_child() if run_manager else None,
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query,
            dialect=self.db.dialect,
            callbacks=run_manager.get_child() if run_manager else None,
        )
