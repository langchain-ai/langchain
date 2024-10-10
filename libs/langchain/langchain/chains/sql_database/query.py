from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS

if TYPE_CHECKING:
    from langchain_community.utilities.sql_database import SQLDatabase


import re


def _strip(text: str) -> str:
    """
    Extracts the pure SQL query from the LLM output.

    Supports the following formats:
    1. Contains a ```sql``` code block
    2. Starts with "SQLQuery:"
    3. Contains only an SQL statement

    If both a ```sql``` code block and an "SQLQuery:" prefix are present, the SQL code block is prioritized.

    Args:
        text (str): The original output from the LLM.

    Returns:
        str: The extracted SQL query.
    """
    # Try to match all ```sql ... ``` code blocks
    code_block_pattern = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    code_blocks = code_block_pattern.findall(text)
    if code_blocks:
        # If multiple code blocks are found, assume the first one is the desired SQL query
        return code_blocks[0].strip()

    # If no code block is found, try to match the "SQLQuery:" prefix
    prefix_pattern = re.compile(r"^SQLQuery:\s*(.*)", re.IGNORECASE | re.DOTALL)
    prefix_match = prefix_pattern.search(text)
    if prefix_match:
        # Extract the content following "SQLQuery:" and try to remove potential code block markers
        sql_part = prefix_match.group(1).strip()
        # Try to extract SQL from a nested code block, if present
        nested_code_match = code_block_pattern.search(sql_part)
        if nested_code_match:
            return nested_code_match.group(1).strip()
        else:
            return sql_part

    # If none of the above matches, assume the entire text is an SQL query
    return text.strip()


class SQLInput(TypedDict):
    """Input for a SQL Chain."""

    question: str


class SQLInputWithTables(TypedDict):
    """Input for a SQL Chain."""

    question: str
    table_names_to_use: List[str]


def create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Union[SQLInput, SQLInputWithTables, Dict[str, Any]], str]:
    """Create a chain that generates SQL queries.

    *Security Note*: This chain generates SQL queries for the given database.

        The SQLDatabase class provides a get_table_info method that can be used
        to get column information as well as sample data from the table.

        To mitigate risk of leaking sensitive data, limit permissions
        to read and scope to the tables that are needed.

        Optionally, use the SQLInputWithTables input type to specify which tables
        are allowed to be accessed.

        Control access to who can submit requests to this chain.

        See https://python.langchain.com/docs/security for more information.

    Args:
        llm: The language model to use.
        db: The SQLDatabase to generate the query for.
        prompt: The prompt to use. If none is provided, will choose one
            based on dialect. Defaults to None. See Prompt section below for more.
        k: The number of results per select statement to return. Defaults to 5.

    Returns:
        A chain that takes in a question and generates a SQL query that answers
        that question.

    Example:

        .. code-block:: python

            # pip install -U langchain langchain-community langchain-openai
            from langchain_openai import ChatOpenAI
            from langchain.chains import create_sql_query_chain
            from langchain_community.utilities import SQLDatabase

            db = SQLDatabase.from_uri("sqlite:///Chinook.db")
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            chain = create_sql_query_chain(llm, db)
            response = chain.invoke({"question": "How many employees are there"})

    Prompt:
        If no prompt is provided, a default prompt is selected based on the SQLDatabase dialect. If one is provided, it must support input variables:
            * input: The user question plus suffix "\nSQLQuery: " is passed here.
            * top_k: The number of results per select statement (the `k` argument to
                this function) is passed in here.
            * table_info: Table definitions and sample rows are passed in here. If the
                user specifies "table_names_to_use" when invoking chain, only those
                will be included. Otherwise, all tables are included.
            * dialect (optional): If dialect input variable is in prompt, the db
                dialect will be passed in here.

        Here's an example prompt:

        .. code-block:: python

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
    """  # noqa: E501
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables)
    ):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }
    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore
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
        | _strip
    )
