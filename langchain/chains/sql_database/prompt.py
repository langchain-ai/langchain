# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

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

__all__ = ("mssql", "mysql", "mariadb", "oracle", "postgresql", "sqlite")

_mssql_system_template = """You are an MS SQL expert. Given an input question, first create a syntactically correct MS SQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the TOP clause as per MS SQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}"""

MSSQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mssql_system_template + "\n\nQuestion: {input}",
)

MSSQL_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_mssql_system_template),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ]
)


_mysql_system_template = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}"""

MYSQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mssql_system_template + "\n\nQuestion: {input}",
)

MYSQL_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_mysql_system_template),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ]
)


_mariadb_system_template = """You are a MariaDB expert. Given an input question, first create a syntactically correct MariaDB query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MariaDB. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}"""

MARIADB_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mssql_system_template + "\n\nQuestion: {input}",
)

MARIADB_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_mariadb_system_template),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ]
)


_oracle_system_template = """You are an Oracle SQL expert. Given an input question, first create a syntactically correct Oracle SQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the FETCH FIRST n ROWS ONLY clause as per Oracle SQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}"""

ORACLE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mssql_system_template + "\n\nQuestion: {input}",
)

ORACLE_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_oracle_system_template),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ]
)


_postgres_system_template = """You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}"""

POSTGRES_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mssql_system_template + "\n\nQuestion: {input}",
)

POSTGRES_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_postgres_system_template),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ]
)


_sqlite_system_template = """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

Question: {input}"""

SQLITE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mssql_system_template + "\n\nQuestion: {input}",
)

SQLITE_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_sqlite_system_template),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ]
)


