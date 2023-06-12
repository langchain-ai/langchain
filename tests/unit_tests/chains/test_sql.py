"""Test SQL functionality."""
import pytest

from langchain import SQLDatabase
from langchain.chains.sql_database.base import (
    SQLDatabaseChain,
    SQLValidation,
    validate_sql,
)
from langchain.chains.sql_database.prompt import SQL_PROMPTS, SQLITE_PROMPT
from tests.unit_tests.llms.fake_llm import FakeLLM

_SQLITE_IN_MEMORY_DB_URI = "sqlite://"

# Arrange
@pytest.fixture
def in_memory_db() -> SQLDatabase:
    database = SQLDatabase.from_uri(_SQLITE_IN_MEMORY_DB_URI)

    # Create table 'users'
    database.run(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE
        )
    """
    )

    # Create table 'orders'
    database.run(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_name TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """
    )

    # Create table 'products'
    database.run(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL
        )
    """
    )

    # Insert some stub data
    [
        database.run(
            f"INSERT INTO users VALUES ({value[0]}, '{value[1]}', '{value[2]}')"
        )
        for value in [
            (1, "John Doe", "john@example.com"),
            (2, "Jane Doe", "jane@example.com"),
        ]
    ]
    [
        database.run(
            f"INSERT INTO orders VALUES ({value[0]}, {value[1]}, '{value[2]}')"
        )
        for value in [(1, 1, "iPhone 13"), (2, 2, "Samsung Galaxy S21")]
    ]
    [
        database.run(
            f"INSERT INTO products VALUES ({value[0]}, '{value[1]}', {value[2]})"
        )
        for value in [(1, "iPhone 13", 999.99), (2, "Samsung Galaxy S21", 799.99)]
    ]

    return database

@pytest.mark.requires("sqlfluff")
def test_simple_question(in_memory_db: SQLDatabase) -> None:
    question = "What is the price of an iphone?"
    prompt = SQLITE_PROMPT.format(top_k=5, input=question, table_info="")
    queries = {
        f"{prompt}\nSQLQuery:": "SELECT price FROM products WHERE name like '%iphone%'",
        f"{prompt}\nSQLQuery:SELECT price FROM products WHERE name like '%iphone%'\nSQLResult: [(999.99,)]\nAnswer:": "999.99\n",
    }
    fake_llm = FakeLLM(queries=queries)
    fake_llm_db_chain = SQLDatabaseChain.from_llm(fake_llm, in_memory_db)
    output = fake_llm_db_chain.run(question)
    assert output == "999.99"

@pytest.mark.requires("sqlfluff")
def test_malformed_sql_query() -> None:
    with pytest.raises(ValueError):
        validate_sql(
            "SELECT price FROM products WHERE name like %iphone%",
            "sqlite",
            SQLValidation(),
        )

@pytest.mark.requires("sqlfluff")
def test_invalid_sql_dialect() -> None:
    with pytest.warns(
        match="Dialect xxxxxxxxxxxxxx unsupported for SQL validation. No validation will be done. Go to https://docs.sqlfluff.com/en/stable/dialects.html to see supported dialects"
    ):
        validate_sql(
            "SELECT price FROM products WHERE name like %iphone%",
            "xxxxxxxxxxxxxx",
            SQLValidation(),
        )
    with pytest.raises(ValueError):
        validate_sql(
            "SELECT price FROM products WHERE name like %iphone%",
            "xxxxxxxxxxxxxx",
            SQLValidation(allow_unsupported_dialect=False),
        )

@pytest.mark.requires("sqlfluff")
def test_non_select_sql_query() -> None:
    validate_sql(
        "DROP TABLE products", "sqlite", SQLValidation(allow_non_select_statements=True)
    )

    with pytest.raises(ValueError):
        validate_sql(
            "DROP TABLE products",
            "sqlite",
            SQLValidation(allow_non_select_statements=False),
        )
    with pytest.raises(ValueError):
        validate_sql(
            "INSERT INTO users (user_id, user_name) VALUES (2, 'Sam');",
            "sqlite",
            SQLValidation(allow_non_select_statements=False),
        )

@pytest.mark.requires("sqlfluff")
def test_select_all_sql_query() -> None:
    validate_sql(
        "SELECT * FROM users;",
        "sqlite",
        SQLValidation(allow_select_all_statements=True),
    )
    with pytest.raises(ValueError):
        validate_sql(
            "SELECT * FROM users;",
            "sqlite",
            SQLValidation(allow_select_all_statements=False),
        )

@pytest.mark.requires("sqlfluff")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parsing_all_supported_dialects():
    sql_statements = [
        "SELECT * FROM users;",
        "SELECT user_id, user_name FROM users;",
        "SELECT * FROM users WHERE user_id = 1;",
        "SELECT COUNT(*) FROM users GROUP BY user_id;",
        "INSERT INTO users (user_id, user_name) VALUES (2, 'Sam');",
    ]
    for sql_stamenet in sql_statements:
        for dialect in SQL_PROMPTS.keys():
            validate_sql(
                sql_stamenet,
                dialect,
                SQLValidation(
                    allow_unsupported_dialect=True, allow_non_select_statements=True
                ),
            )
