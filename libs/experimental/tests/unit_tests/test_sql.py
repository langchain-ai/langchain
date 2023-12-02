from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase

from langchain_experimental.sql.base import SQLDatabaseChain, SQLDatabaseSequentialChain
from tests.unit_tests.fake_llm import FakeLLM

# Fake db to test SQL-Chain
db = SQLDatabase.from_uri("sqlite:///:memory:")


def create_fake_db(db: SQLDatabase) -> SQLDatabase:
    """Create a table in fake db to test SQL-Chain"""
    db.run(
        """
    CREATE TABLE foo (baaz TEXT);
    """
    )
    db.run(
        """
    INSERT INTO foo (baaz)
    VALUES ('baaz');
    """
    )
    return db


db = create_fake_db(db)


def test_sql_chain_without_memory() -> None:
    queries = {"foo": "SELECT baaz from foo", "foo2": "SELECT baaz from foo"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    assert db_chain.run("hello") == "SELECT baaz from foo"


def test_sql_chain_sequential_without_memory() -> None:
    queries = {
        "foo": "SELECT baaz from foo",
        "foo2": "SELECT baaz from foo",
        "foo3": "SELECT baaz from foo",
    }
    llm = FakeLLM(queries=queries, sequential_responses=True)
    db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)
    assert db_chain.run("hello") == "SELECT baaz from foo"


def test_sql_chain_with_memory() -> None:
    valid_prompt_with_history = """
    Only use the following tables:
    {table_info}
    Question: {input}

    Given an input question, first create a syntactically correct
    {dialect} query to run.
    Always limit your query to at most {top_k} results.

    Relevant pieces of previous conversation:
    {history}

    (You do not need to use these pieces of information if not relevant)
    """
    prompt = PromptTemplate(
        input_variables=["input", "table_info", "dialect", "top_k", "history"],
        template=valid_prompt_with_history,
    )
    queries = {"foo": "SELECT baaz from foo", "foo2": "SELECT baaz from foo"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    memory = ConversationBufferMemory()
    db_chain = SQLDatabaseChain.from_llm(
        llm, db, memory=memory, prompt=prompt, verbose=True
    )
    assert db_chain.run("hello") == "SELECT baaz from foo"


def test_sql_chain_sequential_with_memory() -> None:
    valid_query_prompt_str = """
    Only use the following tables:
    {table_info}
    Question: {input}

    Given an input question, first create a syntactically correct
    {dialect} query to run.
    Always limit your query to at most {top_k} results.

    Relevant pieces of previous conversation:
    {history}

    (You do not need to use these pieces of information
    if not relevant)
    """
    valid_decider_prompt_str = """Given the below input question and list of potential
    tables, output a comma separated list of the
    table names that may be necessary to answer this question.

    Question: {query}

    Table Names: {table_names}

    Relevant Table Names:"""

    valid_query_prompt = PromptTemplate(
        input_variables=["input", "table_info", "dialect", "top_k", "history"],
        template=valid_query_prompt_str,
    )
    valid_decider_prompt = PromptTemplate(
        input_variables=["query", "table_names"],
        template=valid_decider_prompt_str,
        output_parser=CommaSeparatedListOutputParser(),
    )
    queries = {
        "foo": "SELECT baaz from foo",
        "foo2": "SELECT baaz from foo",
        "foo3": "SELECT baaz from foo",
    }
    llm = FakeLLM(queries=queries, sequential_responses=True)
    memory = ConversationBufferMemory(memory_key="history", input_key="query")
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm,
        db,
        memory=memory,
        decider_prompt=valid_decider_prompt,
        query_prompt=valid_query_prompt,
        verbose=True,
    )
    assert db_chain.run("hello") == "SELECT baaz from foo"
