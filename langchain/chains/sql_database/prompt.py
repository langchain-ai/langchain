# flake8: noqa

from enum import Enum

from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate

from collections import namedtuple
from typing import Any

PROMPT_SUFFIX = """Only use the following tables:
{table_info}

Question: {input}"""

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "top_k"],
    template=_DEFAULT_TEMPLATE + PROMPT_SUFFIX,
)

_DECIDER_TEMPLATE = """Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.

Question: {query}

Table Names: {table_names}

Relevant Table Names:"""
DECIDER_PROMPT = PromptTemplate(
    input_variables=["query", "table_names"],
    template=_DECIDER_TEMPLATE,
    output_parser=CommaSeparatedListOutputParser(),
)


# Add new SQL dialects here
SqlQueryDialect = namedtuple('SqlQueryDialect', ['key', 'name', 'limit_clause', 'quote_char_name', 'quote_char', 'curr_date_func'])
query_dialects = [
    SqlQueryDialect('create', "CrateDB", "LIMIT", "double quotes", '"', "CURRENT_DATE"),
    SqlQueryDialect('duckdb', "DuckDB", "LIMIT", "double quotes", '"', "today()"),
    SqlQueryDialect('googlesql', "GoogleSQL", "LIMIT", "backticks", "`", "CURRENT_DATE()"),
    SqlQueryDialect('mssql', "MS SQL", "TOP", "square brackets", "[]", "CAST(GETDATE() as date)"),
    SqlQueryDialect('mysql', "MySQL", "LIMIT", "backticks", "`", "CURDATE()"),
    SqlQueryDialect('mariadb', "MariaDB", "LIMIT", "backticks", "`", "CURDATE()"),
    SqlQueryDialect('oracle', "Oracle SQL", "FETCH FIRST n ROWS ONLY", "double quotes", '"', "TRUNC(SYSDATE)"),
    SqlQueryDialect('postgresql', "PostgreSQL", "LIMIT", "double quotes", '"', "CURRENT_DATE"),
    SqlQueryDialect('sqlite', "SQLite", "LIMIT", "double quotes", '"', "date('now')"),
    SqlQueryDialect('clickhouse', "Clickhouse", "LIMIT", "double quotes", '"', "today()"),
    SqlQueryDialect('prestodb', "PrestoDB", "LIMIT", "double quotes", '"', "current_date")
    ]

class SqlQueryDict(dict):
    def __missing__(self, key) -> str:
        return key.join("{}")

def create_prompt(dialect : SqlQueryDialect) -> str:
    base_prompt = """You are a {dialect_name} expert. Given an input question, first create a syntactically 
    correct {dialect_name} query to run, then look at the results of the query and return the answer to the input 
    question. Unless the user specifies in the question a specific number of examples to obtain, query for at 
    most {top_k} results using the {limit_clause} clause as per {dialect_name}. You can order the results to 
    return the most informative data in the database. Never query for all columns from a table. You must query 
    only the columns that are needed to answer the question. Wrap each column name in {quote_char_name} ({quote_char}) 
    to denote them as delimited identifiers. Pay attention to use only the column names you can see 
    in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which 
    column is in which table. Pay attention to use {curr_date_func} function to get the current date, 
    if the question involves "today". 

    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    """

    # Using special dict in order to preserve unused placeholders - e.g. {top_k}
    d = SqlQueryDict(
        {
            "dialect_name": dialect.name,
            "limit_clause": dialect.limit_clause,
            "quote_char_name": dialect.quote_char_name,
            "quote_char": dialect.quote_char,
            "curr_date_func": dialect.curr_date_func
        }
    )
    return base_prompt.format_map(d)


SQL_PROMPTS = {}
for dialect in query_dialects:
    pt = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template=create_prompt(dialect) + PROMPT_SUFFIX)
    SQL_PROMPTS[dialect.key] = pt

