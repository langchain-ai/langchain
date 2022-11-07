# flake8: noqa
from langchain.prompts.prompt import Prompt

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

Question: {input}"""
PROMPT = Prompt(
    input_variables=["input", "table_info", "dialect"],
    template=_DEFAULT_TEMPLATE,
)
