from typing import List, Optional, TypedDict, Union

from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import NoOpOutputParser
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.runnable import RunnableMap, RunnableSequence
from langchain.utilities.sql_database import SQLDatabase


def _strip(text: str) -> str:
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
) -> RunnableSequence[Union[SQLInput, SQLInputWithTables], str]:
    """Create a chain that generates SQL queries.

    Args:
        llm: The language model to use
        db: The SQLDatabase to generate the query for
        prompt: The prompt to use. If none is provided, will choose one
            based on dialect. Defaults to None.
        k: The number of results per select statement to return. Defaults to 5.

    Returns:
        A chain that takes in a question and generates a SQL query that answers
        that question.
    """
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "top_k": lambda _: k,
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }
    if "dialect" in prompt_to_use.input_variables:
        inputs["dialect"] = lambda _: (db.dialect, prompt_to_use)
    return (
        RunnableMap(inputs)
        | prompt_to_use
        | llm.bind(stop=["\nSQLResult:"])
        | NoOpOutputParser()
        | _strip
    )
