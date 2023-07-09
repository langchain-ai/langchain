import pytest

from langchain import SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.sql_database import SQLDatabase
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_sql_chain_without_memory() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    assert db_chain.run("hello") == "Final Answer: baz"


def test_sql_chain_with_valid_memory() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    memory = ConversationBufferMemory(memory_key="history", input_key="input")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    assert db_chain.run("hello") == "Final Answer: baz"


def test_sql_chain_invalid_memory_key() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    with pytest.raises(ValueError):
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
        db_chain = SQLDatabaseChain.from_llm(llm, db, memory=memory, verbose=True)


def test_sql_chain_invalid_memory_input_key() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    with pytest.raises(ValueError):
        memory = ConversationBufferMemory(memory_key="history", input_key="query")
        db_chain = SQLDatabaseChain.from_llm(llm, db, memory=memory, verbose=True)
