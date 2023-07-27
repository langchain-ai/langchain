from typing import Optional

from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import NoOpOutputParser
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableMap
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
    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "top_k": lambda _: k,
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }
    if "dialect" in prompt_to_use.input_variables:
        inputs["dialect"] = lambda _: db.dialect, prompt_to_use
    return (
        RunnableMap(inputs)
        | prompt_to_use
        | llm.bind(stop=["\nSQLResult:"])
        | NoOpOutputParser()
        | _strip
    )
