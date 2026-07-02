from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_classic.chains.sql_database.prompt import PROMPT, SQL_PROMPTS

if TYPE_CHECKING:
    from langchain_community.utilities.sql_database import SQLDatabase


def _strip(text: str) -> str:
    return text.strip()


_FORBIDDEN_SQL_KEYWORDS = (
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
)


def validate_sql_query(sql: str, *, allow_write: bool = False) -> str:
    """Validate a generated SQL query before execution.

    Args:
        sql: SQL query string to validate.
        allow_write: If ``False``, only ``SELECT`` statements are permitted.

    Returns:
        The trimmed SQL string.

    Raises:
        ValueError: If the SQL query fails validation.
    """
    trimmed = sql.strip().rstrip(";")
    if not trimmed:
        msg = "SQL query is empty"
        raise ValueError(msg)
    if ";" in trimmed:
        msg = "Multiple SQL statements are not allowed"
        raise ValueError(msg)

    upper = trimmed.upper()
    if not allow_write and not upper.lstrip().startswith("SELECT"):
        msg = "Only SELECT statements are allowed when allow_write=False"
        raise ValueError(msg)

    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            if allow_write and keyword in {"INSERT", "UPDATE", "DELETE"}:
                continue
            msg = f"SQL statement contains forbidden keyword: {keyword}"
            raise ValueError(msg)

    return trimmed


def _maybe_validate_sql(sql: str, *, validate_sql: bool, allow_write: bool) -> str:
    if validate_sql:
        return validate_sql_query(sql, allow_write=allow_write)
    return _strip(sql)


class SQLInput(TypedDict):
    """Input for a SQL Chain."""

    question: str


class SQLInputWithTables(TypedDict):
    """Input for a SQL Chain."""

    question: str
    table_names_to_use: list[str]


def create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: BasePromptTemplate | None = None,
    k: int = 5,
    *,
    get_col_comments: bool | None = None,
    validate_sql: bool = False,
    allow_write_sql: bool = False,
) -> Runnable[SQLInput | SQLInputWithTables | dict[str, Any], str]:
    r"""Create a chain that generates SQL queries.

    *Security Note*: This chain generates SQL queries for the given database.

        The SQLDatabase class provides a get_table_info method that can be used
        to get column information as well as sample data from the table.

        To mitigate risk of leaking sensitive data, limit permissions
        to read and scope to the tables that are needed.

        Optionally, use the SQLInputWithTables input type to specify which tables
        are allowed to be accessed.

        Control access to who can submit requests to this chain.

        See https://docs.langchain.com/oss/python/security-policy for more information.

    Args:
        llm: The language model to use.
        db: The SQLDatabase to generate the query for.
        prompt: The prompt to use. If none is provided, will choose one
            based on dialect.  See Prompt section below for more.
        k: The number of results per select statement to return.
        get_col_comments: Whether to retrieve column comments along with table info.

    Returns:
        A chain that takes in a question and generates a SQL query that answers
        that question.

    Example:
        ```python
        # pip install -U langchain langchain-community langchain-openai
        from langchain_openai import ChatOpenAI
        from langchain_classic.chains import create_sql_query_chain
        from langchain_community.utilities import SQLDatabase

        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        model = ChatOpenAI(model="gpt-5.5", temperature=0)
        chain = create_sql_query_chain(model, db)
        response = chain.invoke({"question": "How many employees are there"})
        ```

    Prompt:
        If no prompt is provided, a default prompt is selected based on the SQLDatabase
        dialect. If one is provided, it must support input variables:

            * input: The user question plus suffix "\\nSQLQuery: " is passed here.
            * top_k: The number of results per select statement (the `k` argument to
                this function) is passed in here.
            * table_info: Table definitions and sample rows are passed in here. If the
                user specifies "table_names_to_use" when invoking chain, only those
                will be included. Otherwise, all tables are included.
            * dialect (optional): If dialect input variable is in prompt, the db
                dialect will be passed in here.

        Here's an example prompt:

        ```python
        from langchain_core.prompts import PromptTemplate

        template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:

        {table_info}.

        Question: {input}'''
        prompt = PromptTemplate.from_template(template)
        ```
    """  # noqa: E501
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables),
    ):
        msg = (
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
        raise ValueError(msg)
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    table_info_kwargs = {}
    if get_col_comments:
        if db.dialect not in ("postgresql", "mysql", "oracle"):
            msg = (
                f"get_col_comments=True is only supported for dialects "
                f"'postgresql', 'mysql', and 'oracle'. Received dialect: "
                f"{db.dialect}"
            )
            raise ValueError(msg)
        table_info_kwargs["get_col_comments"] = True

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use"),
            **table_info_kwargs,
        ),
    }
    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore[return-value]
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | (lambda sql: _maybe_validate_sql(sql, validate_sql=validate_sql, allow_write=allow_write_sql))
    )
