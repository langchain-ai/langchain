# flake8: noqa
"""Tools for interacting with a SQL database."""

from typing import Any, Dict, Optional, Sequence, Type, Union

from sqlalchemy.engine import Result

from pydantic import BaseModel, Field, root_validator, model_validator, ConfigDict

from langchain_core._api.deprecation import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import (
    QUERY_CHECKER,
    SQL_GENERATION_PROMPT,
)


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QuerySQLDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


class QuerySQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for querying a SQL database.

    .. versionchanged:: 0.3.12

        Renamed from QuerySQLDataBaseTool to QuerySQLDatabaseTool.
        Legacy name still works for backwards compatibility.
    """

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result..
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QuerySQLDatabaseToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)


@deprecated(
    since="0.3.12",
    removal="1.0",
    alternative_import="langchain_community.tools.QuerySQLDatabaseTool",
)
class QuerySQLDataBaseTool(QuerySQLDatabaseTool):  # type: ignore[override]
    """
    Equivalent stub to QuerySQLDatabaseTool for backwards compatibility.
    :private:"""

    ...


class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(
            [t.strip() for t in table_names.split(",")]
        )


class _ListSQLDatabaseToolInput(BaseModel):
    tool_input: str = Field("", description="An empty string")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting tables names."""

    name: str = "sql_db_list_tables"
    description: str = "Input is an empty string, output is a comma-separated list of tables in the database."
    args_schema: Type[BaseModel] = _ListSQLDatabaseToolInput

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a comma-separated list of table names."""
        return ", ".join(self.db.get_usable_table_names())


class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed and SQL query to be checked.")


class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    @model_validator(mode="before")
    @classmethod
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Any:
        if "llm_chain" not in values:
            from langchain.chains.llm import LLMChain

            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),  # type: ignore[arg-type]
                prompt=PromptTemplate(
                    template=QUERY_CHECKER,
                    input_variables=[
                        "dialect",
                        "query",
                        "table_info_str",
                        "foreign_key_info_str",
                    ],
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
            table_info_str=self.db.table_info_str,
            foreign_key_info_str=self.db.foreign_key_info_str,
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


class _GenerateSQLToolInput(BaseModel):
    question: str = Field(
        ..., description="The question to be converted into a SQL query."
    )


class QuerySQLGeneratorTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for generating SQL queries from natural language questions."""

    name: str = "sql_db_query_generator"
    description: str = """
    Convert a natural language question into a SQL query.
    This tool will analyze the database schema and relationships to generate an appropriate query.
    The generated query should be checked with sql_db_query_checker before execution.
    """
    args_schema: Type[BaseModel] = _GenerateSQLToolInput
    llm: BaseLanguageModel = Field(..., exclude=True)

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate a SQL query from a natural language question."""
        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate

        # Create the generation prompt
        prompt = PromptTemplate(
            template=SQL_GENERATION_PROMPT,
            input_variables=[
                "table_info_str",
                "foreign_key_info_str",
                "sample_rows",
                "question",
            ],
        )

        # Initialize the chain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Get database context
        context = {
            "table_info_str": self.db.table_info_str,
            "foreign_key_info_str": self.db.foreign_key_info_str,
            "sample_rows": self.db.get_sample_rows_str(3),
            "question": question,
        }

        # Generate the query
        return chain.predict(**context)

    async def _arun(
        self,
        question: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of query generation."""
        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            template=SQL_GENERATION_PROMPT,
            input_variables=[
                "table_info_str",
                "foreign_key_info_str",
                "sample_rows",
                "question",
            ],
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        context = {
            "table_info_str": self.db.table_info_str,
            "foreign_key_info_str": self.db.foreign_key_info_str,
            "sample_rows": self.db.get_sample_rows_str(3),
            "question": question,
        }

        return await chain.apredict(**context)
