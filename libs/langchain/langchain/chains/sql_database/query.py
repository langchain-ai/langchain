from typing import Optional

from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.chat_models.base import BaseChatModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import NoOpOutputParser
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.utilities.sql_database import SQLDatabase


def _strip(text: str) -> str:
    return text.strip()


def create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> RunnableSequence:
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    formatted_prompt = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "dialect": lambda _: db.dialect,
        "top_k": lambda _: k,
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    } | prompt_to_use
    return (
        formatted_prompt | llm.bind(stop=["\nSQLResult:"]) | NoOpOutputParser() | _strip
    )
