# flake8: noqa
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate


_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the tables listed below.

{table_info}

Question: {input}"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "top_k"],
    template=_DEFAULT_TEMPLATE,
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

_meta_prompt = """You are a {dialect} expert. Given an input question, let's think step by step. Each step, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {{top_k}} results using the {limit_clause} clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in {quote} to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{{table_info}}

Question: {{input}}"""

DUCKDB_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="DuckDB", limit_clause="LIMIT", quote='double quotes (")'
    ),
)

GOOGLESQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="GoogleSQL", limit_clause="LIMIT", quote="backticks (`)"
    ),
)

MSSQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="MS SQL", limit_clause="TOP", quote="square brackets ([])"
    ),
)

MYSQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="MySQL", limit_clause="LIMIT", quote="backticks (`)"
    ),
)

MARIADB_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="MariaDB", limit_clause="LIMIT", quote="backticks (`)"
    ),
)

ORACLE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="Oracle SQL",
        limit_clause="FETCH FIRST n ROWS ONLY",
        quote='double quotes (")',
    ),
)

POSTGRES_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="PostgreSQL", limit_clause="LIMIT", quote="backticks (`)"
    ),
)

SQLITE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_meta_prompt.format(
        dialect="SQLite", limit_clause="LIMIT", quote='double quotes (")'
    ),
)


SQL_PROMPTS = {
    "duckdb": DUCKDB_PROMPT,
    "googlesql": GOOGLESQL_PROMPT,
    "mssql": MSSQL_PROMPT,
    "mysql": MYSQL_PROMPT,
    "mariadb": MARIADB_PROMPT,
    "oracle": ORACLE_PROMPT,
    "postgresql": POSTGRES_PROMPT,
    "sqlite": SQLITE_PROMPT,
}
