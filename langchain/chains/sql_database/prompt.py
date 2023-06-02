# flake8: noqa

from enum import Enum

from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate

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


class SqlQueryDict(dict):
    def __missing__(self, key):
        return key.join("{}")


class SqlQueryDialect(Enum):
    """
    Encapsulates variations in syntax for SQL query prompts. For the purposes of SQL query
    prompts, all variations (currently) are described by -  quote character, row limiting clause,
    and current date function. To add a new SQL dialect simply add a new enumerated value - e.g. FOODB = (...).
    """

    DUCKDB = ("DuckDB", "LIMIT", "double quotes", '"', "today()")
    GOOGLESQL = ("GoogleSQL", "LIMIT", "backticks", "`", "CURRENT_DATE()")
    MSSQL = ("MS SQL", "TOP", "square brackets", "[]", "CAST(GETDATE() as date)")
    MYSQL = ("MySQL", "LIMIT", "backticks", "`", "CURDATE()")
    MARIADB = ("MariaDB", "LIMIT", "backticks", "`", "CURDATE()")
    ORACLE = (
        "Oracle SQL",
        "FETCH FIRST n ROWS ONLY",
        "double quotes",
        '"',
        "TRUNC(SYSDATE)",
    )

    POSTGRES = ("PostgreSQL", "LIMIT", "double quotes", '"', "CURRENT_DATE")
    SQLITE = ("SQLite", "LIMIT", "double quotes", '"', "date('now')")
    CLICKHOUSE = ("Clickhouse", "LIMIT", "double quotes", '"', "today()")
    PRESTODB = ("PrestoDB", "LIMIT", "double quotes", '"', "current_date")

    def __init__(
        self,
        dialect_name: str,
        limit_clause: str,
        quote_char_name: str,
        quote_char: str,
        curr_date_func: str,
    ):
        self.dialect_name = dialect_name
        self.limit_clause = limit_clause
        self.quote_char_name = quote_char_name
        self.quote_char = quote_char
        self.curr_date_func = curr_date_func

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}: {self.dialect_name}, {self.limit_clause}, {self.quote_char_name}, {self.quote_char}, {self.curr_date_func}"

    def create_prompt(self) -> str:
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

        # make sure to preserve unused placeholders - e.g. {top_k}
        d = SqlQueryDict(
            {
                "dialect_name": self.dialect_name,
                "limit_clause": self.limit_clause,
                "quote_char_name": self.quote_char_name,
                "quote_char": self.quote_char,
                "curr_date_func": self.curr_date_func,
            }
        )
        return base_prompt.format_map(d)


_duckdb_prompt = SqlQueryDialect.DUCKDB.create_prompt()

DUCKDB_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_duckdb_prompt + PROMPT_SUFFIX,
)

_googlesql_prompt = SqlQueryDialect.GOOGLESQL.create_prompt()

GOOGLESQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_googlesql_prompt + PROMPT_SUFFIX,
)

_mssql_prompt = SqlQueryDialect.MSSQL.create_prompt()

MSSQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mssql_prompt + PROMPT_SUFFIX,
)

_mysql_prompt = SqlQueryDialect.MYSQL.create_prompt()

MYSQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mysql_prompt + PROMPT_SUFFIX,
)

_mariadb_prompt = SqlQueryDialect.MARIADB.create_prompt()

MARIADB_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mariadb_prompt + PROMPT_SUFFIX,
)

_oracle_prompt = SqlQueryDialect.ORACLE.create_prompt()

ORACLE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_oracle_prompt + PROMPT_SUFFIX,
)

_postgres_prompt = SqlQueryDialect.POSTGRES.create_prompt()

POSTGRES_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_postgres_prompt + PROMPT_SUFFIX,
)

_sqlite_prompt = SqlQueryDialect.SQLITE.create_prompt()

SQLITE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_sqlite_prompt + PROMPT_SUFFIX,
)

_clickhouse_prompt = SqlQueryDialect.CLICKHOUSE.create_prompt()

CLICKHOUSE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_clickhouse_prompt + PROMPT_SUFFIX,
)

_prestodb_prompt = SqlQueryDialect.PRESTODB.create_prompt()

PRESTODB_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_prestodb_prompt + PROMPT_SUFFIX,
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
    "clickhouse": CLICKHOUSE_PROMPT,
    "prestodb": PRESTODB_PROMPT,
}
