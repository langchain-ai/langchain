# flake8: noqa
from langchain.prompt import Prompt

_DEFAULT_TEMPLATE = """Given an input question, respond with syntactically correct {dialect}. Be creative but the SQL must be correct. Only use the following tables:

{table_info}

Input: {input}"""
PROMPT = Prompt(
    input_variables=["input", "table_info", "dialect"],
    template=_DEFAULT_TEMPLATE,
)
